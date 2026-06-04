"""
Preprocessing pipeline for eSFM: Enzyme-Substrate binding specificity.

Loads ReactZyme (Hua et al., NeurIPS 2024) enzyme-reaction pairs, extracts
the primary substrate SMILES (filtering cofactors like ATP, NAD+, water),
and embeds enzymes with ESM-2 and substrates with MoLFormer-XL.

Architecture note: eSFM is the REVERSE of dtSFM.
  - dtSFM: agent = drug (MoLFormer), target = protein (ESM-2)
  - eSFM:  agent = enzyme protein (ESM-2), target = substrate (MoLFormer)
But in CALM's config system, model_ag/model_ab refer to the first/second
encoder slot. To reuse dtSFM's proven config pattern:
  ag = substrate SMILES (MoLFormer, 768-dim)  — same slot as "drug" in dtSFM
  ab = enzyme protein  (ESM-2, 1280-dim)      — same slot as "protein" in dtSFM

Data source:
  ReactZyme — 178,463 enzyme-reaction pairs from SwissProt + Rhea
  Download: https://zenodo.org/records/11494913

Usage:
    # Step 1: Build metadata from ReactZyme .pt splits (Mac or Euler login node)
    python -m calm.preprocess.esfm download --output_dir data/esfm

    # Step 2: Embed enzymes + substrates (Euler GPU job)
    python -m calm.preprocess.esfm embed --output_dir data/esfm --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 1. Cofactor filtering
# ---------------------------------------------------------------------------

# Exact SMILES strings for common small cofactors / ions / solvent
COFACTOR_EXACT = {
    "[H]O[H]",             # water
    "[H+]",                # proton
    "O=O",                 # molecular oxygen
    "[Fe+2]",              # iron(II)
    "[Fe+3]",              # iron(III)
    "O=P([O-])([O-])O",   # inorganic phosphate (Pi)
    "O=P([O-])([O-])OP(=O)([O-])[O-]",  # pyrophosphate (PPi)
    "O=C=O",               # carbon dioxide
    "[NH4+]",              # ammonium
    "O",                   # water (alt notation)
    "[O-][O-]",            # peroxide
    "[H][H]",              # H2
}

# Substrings that identify large cofactors (ATP, NAD+, SAM, CoA, FAD, etc.)
COFACTOR_SUBSTRINGS = [
    "OP(=O)([O-])OP(=O)([O-])OP(",              # ATP / GTP (triphosphate)
    "C[S+](CC[C@H]([NH3+])C(=O)[O-])",          # SAM (S-adenosylmethionine)
    "NC(=O)C1=CC=C[N+]",                         # NAD+ / NADH / NADP+ / NADPH
    "S1[Fe]S[Fe+]1",                              # iron-sulfur cluster
    "CC(=O)SCCNC(=O)CCNC(=O)[C@H](O)C(C)(C)COP",  # Coenzyme A / acetyl-CoA
]


def is_cofactor(smi: str) -> bool:
    """Return True if a SMILES fragment is a known cofactor or very small ion."""
    smi = smi.strip()
    if smi in COFACTOR_EXACT:
        return True
    for substr in COFACTOR_SUBSTRINGS:
        if substr in smi:
            return True
    # Very small molecules (rough heuristic: SMILES < 6 chars ~ < 4 heavy atoms)
    if len(smi) < 6:
        return True
    return False


def extract_main_substrate(substrate_smiles: str) -> str | None:
    """Extract the primary substrate from a dot-separated SMILES string.

    ReactZyme stores ALL substrate molecules for a reaction concatenated
    with dots (SMILES notation for disconnected fragments). Most reactions
    have 2-3 substrate molecules: the "real" substrate plus cofactors like
    ATP, water, or NAD+.

    Strategy: remove known cofactors, return the largest remaining molecule
    (by SMILES string length, a rough proxy for molecular weight).

    Returns None if all molecules are cofactors.
    """
    parts = substrate_smiles.split(".")
    filtered = [p for p in parts if not is_cofactor(p)]
    if not filtered:
        return None
    return max(filtered, key=len)


# ---------------------------------------------------------------------------
# 2. Load ReactZyme .pt splits and build metadata
# ---------------------------------------------------------------------------


def load_reactzyme_splits(data_dir: Path) -> dict:
    """Load the ReactZyme pre-built .pt split files.

    ReactZyme provides two split strategies:
      - enzyme-similarity (seq_smi): enzymes clustered by sequence similarity
      - time-based (time): train on older entries, test on newer

    Each .pt file is a dict {int: (substrate_smiles, enzyme_sequence)}.

    Returns dict with keys 'enzyme_sim' and 'time', each containing
    'train_val' and 'test' lists of (substrate, enzyme) tuples.
    """
    splits = {}

    # Enzyme-similarity split
    tv_path = data_dir / "positive_train_val_seq_smi.pt"
    te_path = data_dir / "positive_test_seq_smi.pt"
    if tv_path.exists() and te_path.exists():
        tv = torch.load(tv_path, weights_only=False)
        te = torch.load(te_path, weights_only=False)
        splits["enzyme_sim"] = {
            "train_val": list(tv.values()),
            "test": list(te.values()),
        }
        print(f"  Enzyme-similarity split: {len(tv):,} train+val, {len(te):,} test")

    # Time-based split
    tv_path_t = data_dir / "positive_train_val_time.pt"
    te_path_t = data_dir / "positive_test_time.pt"
    if tv_path_t.exists() and te_path_t.exists():
        tv_t = torch.load(tv_path_t, weights_only=False)
        te_t = torch.load(te_path_t, weights_only=False)
        splits["time"] = {
            "train_val": list(tv_t.values()),
            "test": list(te_t.values()),
        }
        print(f"  Time-based split: {len(tv_t):,} train+val, {len(te_t):,} test")

    return splits


def build_pairs_from_splits(splits: dict) -> pd.DataFrame:
    """Combine all pairs from the enzyme-similarity split, apply cofactor
    filtering, and build a deduplicated DataFrame.

    We use the enzyme-similarity split as the primary dataset because it
    maps to our MMseqs2 OOD protocol. The time-based split will be used
    separately for temporal validation.
    """
    # Use enzyme-similarity split as primary
    all_tuples = splits["enzyme_sim"]["train_val"] + splits["enzyme_sim"]["test"]
    print(f"  Total raw pairs: {len(all_tuples):,}")

    rows = []
    skipped_cofactor = 0
    for substrate_raw, enzyme_seq in all_tuples:
        main_sub = extract_main_substrate(substrate_raw)
        if main_sub is None:
            skipped_cofactor += 1
            continue
        rows.append({
            "enzyme_seq": enzyme_seq,
            "substrate_smiles": main_sub,
            "substrate_raw": substrate_raw,
        })

    print(f"  Skipped (all cofactors): {skipped_cofactor:,}")

    df = pd.DataFrame(rows)

    # Deduplicate on (enzyme_seq, substrate_smiles)
    before = len(df)
    df = df.drop_duplicates(subset=["enzyme_seq", "substrate_smiles"], keep="first")
    print(f"  After dedup: {before:,} -> {len(df):,} unique enzyme-substrate pairs")
    print(f"  Unique enzymes: {df.enzyme_seq.nunique():,}")
    print(f"  Unique substrates: {df.substrate_smiles.nunique():,}")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Build metadata in CALM format
# ---------------------------------------------------------------------------


def build_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Build metadata DataFrame in CALM format.

    CALM naming convention (inherited from antibody-antigen origin):
      ag = "agent" slot = substrate SMILES (MoLFormer)
      ab = "antibody" slot = enzyme protein (ESM-2)

    This matches dtSFM where ag=drug, ab=protein.
    """
    # Map unique substrates to integer IDs
    unique_substrates = df["substrate_smiles"].unique()
    sub_to_id = {s: i for i, s in enumerate(unique_substrates)}

    # Map unique enzyme sequences to integer IDs
    unique_enzymes = df["enzyme_seq"].unique()
    enz_to_id = {s: i for i, s in enumerate(unique_enzymes)}

    df = df.copy()
    df["pair_index"] = range(len(df))
    df["ag_id"] = df["substrate_smiles"].map(sub_to_id)
    df["ab_id"] = df["enzyme_seq"].map(enz_to_id)

    # cluster_id = enzyme sequence identity (for OOD splitting)
    # We use ab_id here as a placeholder; MMseqs2 will generate proper
    # sequence-similarity clusters later
    df["cluster_id"] = df["ab_id"]

    # Unique hash for split indexing (CALM convention)
    df["Unique_ag_vh_vl_hash"] = (
        "enz_" + df["ab_id"].astype(str)
        + "_sub_" + df["ag_id"].astype(str)
        + "_" + df["pair_index"].astype(str)
    )

    # Keep protein_seq column for MMseqs2 clustering
    df["protein_seq"] = df["enzyme_seq"]

    print(f"  Metadata: {len(df):,} pairs, "
          f"{df.ag_id.nunique():,} substrates (ag), "
          f"{df.ab_id.nunique():,} enzymes (ab), "
          f"{df.cluster_id.nunique():,} clusters")

    return df


# ---------------------------------------------------------------------------
# 4. Build time-based split index
# ---------------------------------------------------------------------------


def build_time_split_index(
    metadata: pd.DataFrame,
    splits: dict,
    output_dir: Path,
) -> None:
    """Build a split index file from ReactZyme's time-based split.

    This maps the pre-built time split back to our metadata rows so we can
    use it for temporal validation (analogous to JASPAR 2022→2026 in tSFM).
    """
    if "time" not in splits:
        print("  Time split not available, skipping.")
        return

    # Build lookup: (enzyme_seq, main_substrate) -> hash
    pair_to_hash = {}
    for _, row in metadata.iterrows():
        key = (row["enzyme_seq"], row["substrate_smiles"])
        pair_to_hash[key] = row["Unique_ag_vh_vl_hash"]

    train_hashes = []
    test_hashes = []

    for substrate_raw, enzyme_seq in splits["time"]["train_val"]:
        main_sub = extract_main_substrate(substrate_raw)
        if main_sub is None:
            continue
        key = (enzyme_seq, main_sub)
        if key in pair_to_hash:
            train_hashes.append(pair_to_hash[key])

    for substrate_raw, enzyme_seq in splits["time"]["test"]:
        main_sub = extract_main_substrate(substrate_raw)
        if main_sub is None:
            continue
        key = (enzyme_seq, main_sub)
        if key in pair_to_hash:
            test_hashes.append(pair_to_hash[key])

    # Save as a split JSON (single fold, no inner fold structure)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_data = {
        "train": train_hashes,
        "val": [],  # Will be carved from train during training
        "test": test_hashes,
    }
    out_path = output_dir / "time_split.json"
    with open(out_path, "w") as f:
        json.dump(split_data, f)
    print(f"  Time split: {len(train_hashes):,} train, {len(test_hashes):,} test")
    print(f"  Saved to {out_path}")


# ---------------------------------------------------------------------------
# 5. Embed substrates with MoLFormer-XL
# ---------------------------------------------------------------------------


def embed_substrates_molformer(
    smiles_list: list[str],
    model_name: str = "ibm/MoLFormer-XL-both-10pct",
    device: str = "cpu",
    batch_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Embed substrate SMILES strings using MoLFormer-XL, with mean-pooling.

    Mean-pools during embedding for consistency with the ESM-2 enzyme
    embeddings (which must be mean-pooled at embedding time due to the
    177K sequence count). Result: (N, 1, 768) tensor.

    Parameters
    ----------
    smiles_list : list[str]
        SMILES strings for each substrate.
    model_name : str
        HuggingFace model name.
    device : str
        Device for inference.
    batch_size : int
        Batch size for inference.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        embeddings: (N, 1, 768) — mean-pooled, masks: (N, 1) all True
    """
    from transformers import AutoModel, AutoTokenizer

    print(f"  Loading MoLFormer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name, deterministic_eval=True, trust_remote_code=True
    )
    model = model.to(device)
    model.eval()

    embed_dim = 768
    N = len(smiles_list)
    pooled = torch.zeros(N, 1, embed_dim)

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = smiles_list[start:end]

            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            if isinstance(outputs, tuple):
                hidden_states = outputs[0]
            else:
                hidden_states = outputs.last_hidden_state

            attention_mask = inputs["attention_mask"]

            for j in range(hidden_states.shape[0]):
                valid_len = int(attention_mask[j].sum().item())
                emb = hidden_states[j, :valid_len, :].mean(dim=0)
                pooled[start + j, 0, :] = emb.cpu()

            if end % 200 == 0 or end == N:
                print(f"  MoLFormer: embedded {end}/{N} substrates")

    masks = torch.ones(N, 1, dtype=torch.bool)
    print(f"  MoLFormer embeddings: {pooled.shape}, masks: {masks.shape}")
    return pooled, masks


# ---------------------------------------------------------------------------
# 6. Embed enzymes with ESM-2
# ---------------------------------------------------------------------------


def embed_enzymes_esm2(
    sequences: list[str],
    output_dir: Path,
    model_name: str = "esm2_t33_650M_UR50D",
    device: str = "cpu",
    batch_size: int = 4,
    max_seq_len: int = 1024,
    chunk_size: int = 10000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Embed enzyme protein sequences using ESM-2, with pre-pooling.

    With 177K enzymes, storing token-level embeddings (N × L × 1280) is
    impossible — it would be ~920 GB. Instead, we mean-pool during
    embedding to produce (N, 1, 1280) vectors. CALM's mean-pooling layer
    then becomes a no-op (mean of a single token = that token), so the
    result is mathematically identical.

    We process in chunks saved to disk to keep RAM usage manageable.

    Parameters
    ----------
    sequences : list[str]
        Protein sequences.
    output_dir : Path
        Directory for temporary chunk files.
    model_name : str
        ESM-2 model name.
    device : str
        Device for inference.
    batch_size : int
        Batch size (keep small — ESM-2 is large, enzymes can be long).
    max_seq_len : int
        Maximum sequence length (truncate longer sequences).
    chunk_size : int
        Number of sequences per chunk (saved to disk to free RAM).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        embeddings: (N, 1, 1280) — mean-pooled, masks: (N, 1) all True
    """
    import esm
    import gc

    sequences_trunc = [seq[:max_seq_len] for seq in sequences]
    n_truncated = sum(1 for s in sequences if len(s) > max_seq_len)
    if n_truncated > 0:
        print(f"  Truncated {n_truncated}/{len(sequences)} sequences "
              f"to {max_seq_len} residues")

    print(f"  Loading ESM-2: {model_name}...")
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    chunk_dir = output_dir / "esm2_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    embed_dim = 1280

    n_chunks = (len(sequences_trunc) + chunk_size - 1) // chunk_size
    print(f"  Processing {len(sequences_trunc):,} enzymes in "
          f"{n_chunks} chunks of {chunk_size}")
    print(f"  Using mean-pooling during embedding (saves ~900 GB RAM)")

    chunk_files = []

    with torch.no_grad():
        for chunk_idx in range(n_chunks):
            c_start = chunk_idx * chunk_size
            c_end = min(c_start + chunk_size, len(sequences_trunc))
            chunk_seqs = sequences_trunc[c_start:c_end]
            n_in_chunk = len(chunk_seqs)

            # Store mean-pooled: (n_in_chunk, 1, 1280)
            chunk_embed = torch.zeros(n_in_chunk, 1, embed_dim)

            for start in range(0, n_in_chunk, batch_size):
                end = min(start + batch_size, n_in_chunk)
                batch_seqs = [
                    (f"seq_{i}", seq)
                    for i, seq in enumerate(chunk_seqs[start:end])
                ]
                _, _, batch_tokens = batch_converter(batch_seqs)
                batch_tokens = batch_tokens.to(device)

                results = model(
                    batch_tokens, repr_layers=[33], return_contacts=False
                )
                token_reps = results["representations"][33]

                for j in range(token_reps.shape[0]):
                    seq_len = len(chunk_seqs[start + j])
                    # Mean-pool over valid token positions
                    emb = token_reps[j, 1 : seq_len + 1, :].mean(dim=0)
                    chunk_embed[start + j, 0, :] = emb.cpu()

                global_end = c_start + end
                if global_end % 500 == 0 or global_end == len(sequences_trunc):
                    print(f"  ESM-2: embedded {global_end}/{len(sequences_trunc)} "
                          f"enzymes")

            # Save chunk to disk and free memory
            chunk_path = chunk_dir / f"chunk_{chunk_idx:04d}.pt"
            torch.save(chunk_embed, chunk_path)
            chunk_files.append(chunk_path)
            del chunk_embed
            gc.collect()
            print(f"  Saved chunk {chunk_idx + 1}/{n_chunks} to {chunk_path.name}")

    # --- Assemble final tensor from chunks ---
    print(f"  Assembling {len(chunk_files)} chunks into final tensor...")
    N = len(sequences_trunc)
    padded = torch.zeros(N, 1, embed_dim)

    offset = 0
    for chunk_path in chunk_files:
        data = torch.load(chunk_path, weights_only=True)
        n = data.shape[0]
        padded[offset : offset + n] = data
        offset += n
        del data
        gc.collect()

    # Mask is all True (every sequence has exactly 1 pooled token)
    masks = torch.ones(N, 1, dtype=torch.bool)

    # Clean up chunk files
    for chunk_path in chunk_files:
        chunk_path.unlink()
    chunk_dir.rmdir()

    print(f"  ESM-2 embeddings: {padded.shape}, masks: {masks.shape}")
    print(f"  Final tensor size: {padded.nelement() * 4 / 1e9:.2f} GB")
    return padded, masks


# ---------------------------------------------------------------------------
# 7. Main pipeline: download subcommand
# ---------------------------------------------------------------------------


def run_download(output_dir: str) -> None:
    """Build metadata from ReactZyme .pt splits (no GPU needed).

    Run this on your Mac or on the Euler login node.

    Prerequisites: ReactZyme files must be downloaded from Zenodo into
    output_dir. Required files:
      - positive_train_val_seq_smi.pt  (enzyme-similarity split, train+val)
      - positive_test_seq_smi.pt       (enzyme-similarity split, test)
      - positive_train_val_time.pt     (time split, train+val)
      - positive_test_time.pt          (time split, test)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Step 1: Loading ReactZyme splits...")
    splits = load_reactzyme_splits(out)

    if "enzyme_sim" not in splits:
        raise FileNotFoundError(
            f"ReactZyme split files not found in {out}. "
            "Download them from https://zenodo.org/records/11494913"
        )

    print("\nStep 2: Extracting substrates and filtering cofactors...")
    df = build_pairs_from_splits(splits)

    print("\nStep 3: Building metadata...")
    metadata = build_metadata(df)
    metadata.to_csv(out / "metadata.csv", index=False)

    print("\nStep 4: Building time-based split index...")
    build_time_split_index(metadata, splits, out / "split_index")

    # Summary
    n_enz = metadata.ab_id.nunique()
    n_sub = metadata.ag_id.nunique()
    print(f"\n{'='*60}")
    print(f"Download complete! Saved to {out}")
    print(f"  metadata.csv:     {len(metadata):,} enzyme-substrate pairs")
    print(f"  Unique enzymes:   {n_enz:,} (will embed with ESM-2)")
    print(f"  Unique substrates:{n_sub:,} (will embed with MoLFormer)")
    print(f"  split_index/:     time-based split")
    print(f"{'='*60}")
    print(f"\nNext step: run 'embed' subcommand on Euler GPU node")
    print(f"  WARNING: {n_enz:,} enzyme embeddings will be LARGE (~50-100 GB).")
    print(f"  Use /cluster/scratch/ on Euler for output_dir.")


# ---------------------------------------------------------------------------
# 8. Main pipeline: embed subcommand
# ---------------------------------------------------------------------------


def run_embed(
    output_dir: str,
    device: str = "cpu",
    molformer_batch_size: int = 32,
    esm_batch_size: int = 4,
) -> None:
    """Embed substrates and enzymes (requires GPU).

    Run this on Euler with a GPU allocation.
    """
    out = Path(output_dir)
    metadata_path = out / "metadata.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"metadata.csv not found in {out}. Run 'download' first."
        )

    df = pd.read_csv(metadata_path)
    print(f"Loaded metadata: {len(df):,} pairs")

    # --- Embed unique substrates with MoLFormer (ag slot) ---
    unique_substrates = df["substrate_smiles"].drop_duplicates().tolist()
    sub_to_idx = {s: i for i, s in enumerate(unique_substrates)}
    ag_indices = torch.tensor(
        [sub_to_idx[s] for s in df["substrate_smiles"]], dtype=torch.long
    )

    print(f"\nStep 1: Embedding {len(unique_substrates):,} unique substrates "
          f"with MoLFormer (from {len(df):,} total pairs)...")
    ag_embed, ag_mask = embed_substrates_molformer(
        unique_substrates, device=device, batch_size=molformer_batch_size
    )
    torch.save(ag_embed, out / "ag_embed.pt")
    torch.save(ag_mask, out / "ag_mask.pt")
    torch.save(ag_indices, out / "ag_indices.pt")
    print(f"  Saved ag_embed.pt {ag_embed.shape}, ag_mask.pt, ag_indices.pt")

    # --- Embed unique enzymes with ESM-2 (ab slot) ---
    unique_enzymes = df["enzyme_seq"].drop_duplicates().tolist()
    enz_to_idx = {s: i for i, s in enumerate(unique_enzymes)}
    ab_indices = torch.tensor(
        [enz_to_idx[s] for s in df["enzyme_seq"]], dtype=torch.long
    )

    print(f"\nStep 2: Embedding {len(unique_enzymes):,} unique enzymes "
          f"with ESM-2 (from {len(df):,} total pairs)...")
    ab_embed, ab_mask = embed_enzymes_esm2(
        unique_enzymes, output_dir=out, device=device, batch_size=esm_batch_size
    )
    torch.save(ab_embed, out / "ab_embed.pt")
    torch.save(ab_mask, out / "ab_mask.pt")
    torch.save(ab_indices, out / "ab_indices.pt")
    print(f"  Saved ab_embed.pt {ab_embed.shape}, ab_mask.pt, ab_indices.pt")

    print(f"\nEmbedding complete! All tensors saved to {out}")
    print(f"  ag_embed.pt:     Substrate embeddings (MoLFormer, 768-dim)")
    print(f"  ab_embed.pt:     Enzyme embeddings (ESM-2, 1280-dim)")
    print(f"  ag_indices.pt:   Maps each row -> unique substrate index")
    print(f"  ab_indices.pt:   Maps each row -> unique enzyme index")
    print(f"  ag_mask.pt:      Substrate padding masks")
    print(f"  ab_mask.pt:      Enzyme padding masks")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess ReactZyme data for eSFM training"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- download subcommand ---
    dl = subparsers.add_parser(
        "download",
        help="Build metadata from ReactZyme splits (no GPU needed)",
    )
    dl.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory containing ReactZyme .pt files",
    )

    # --- embed subcommand ---
    em = subparsers.add_parser(
        "embed",
        help="Embed substrates (MoLFormer) and enzymes (ESM-2) — requires GPU",
    )
    em.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory with metadata.csv from download step",
    )
    em.add_argument(
        "--device", type=str, default="cpu",
        help="Device for models (cpu or cuda)",
    )
    em.add_argument(
        "--molformer_batch_size", type=int, default=32,
        help="Batch size for MoLFormer inference",
    )
    em.add_argument(
        "--esm_batch_size", type=int, default=4,
        help="Batch size for ESM-2 inference",
    )

    args = parser.parse_args()

    if args.command == "download":
        run_download(output_dir=args.output_dir)
    elif args.command == "embed":
        run_embed(
            output_dir=args.output_dir,
            device=args.device,
            molformer_batch_size=args.molformer_batch_size,
            esm_batch_size=args.esm_batch_size,
        )
