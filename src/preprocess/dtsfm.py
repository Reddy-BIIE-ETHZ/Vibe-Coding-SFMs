"""
Preprocessing pipeline for dtSFM: Drug-Target (kinase inhibitor) binding.

Combines DAVIS and KIBA drug-target interaction datasets, filters for
positive binding pairs, embeds drugs with MoLFormer-XL and proteins with
ESM-2, and saves CALM-ready tensors.

Data sources:
  - DAVIS: 68 kinase inhibitors × 379 kinases, full Kd selectivity matrix
    (Kd in nM; 10000 = non-binder)
  - KIBA: 2,068 compounds × 229 kinases, integrated bioactivity score
    (lower = stronger binding; threshold ≤ 12.1 for active)

Usage:
    # Step 1: Download data (Mac Terminal, no GPU needed)
    python -m calm.preprocess.dtsfm download --output_dir data/dtsfm

    # Step 2: Embed + build tensors (Euler GPU job)
    python -m calm.preprocess.dtsfm embed --output_dir data/dtsfm --device cuda
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
# 1. Download DAVIS + KIBA from TDC
# ---------------------------------------------------------------------------


def download_datasets(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download DAVIS and KIBA datasets from Therapeutics Data Commons.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        DAVIS and KIBA raw DataFrames.
    """
    from tdc.multi_pred import DTI

    davis_path = output_dir / "davis_raw.csv"
    kiba_path = output_dir / "kiba_raw.csv"

    if davis_path.exists():
        print("  DAVIS already downloaded, loading from cache...")
        df_davis = pd.read_csv(davis_path)
    else:
        print("  Downloading DAVIS from TDC...")
        davis = DTI(name="DAVIS")
        df_davis = davis.get_data()
        df_davis.to_csv(davis_path, index=False)

    if kiba_path.exists():
        print("  KIBA already downloaded, loading from cache...")
        df_kiba = pd.read_csv(kiba_path)
    else:
        print("  Downloading KIBA from TDC...")
        kiba = DTI(name="KIBA")
        df_kiba = kiba.get_data()
        df_kiba.to_csv(kiba_path, index=False)

    print(f"  DAVIS: {len(df_davis):,} pairs, {df_davis.Drug_ID.nunique()} drugs, "
          f"{df_davis.Target_ID.nunique()} targets")
    print(f"  KIBA:  {len(df_kiba):,} pairs, {df_kiba.Drug_ID.nunique()} drugs, "
          f"{df_kiba.Target_ID.nunique()} targets")

    return df_davis, df_kiba


# ---------------------------------------------------------------------------
# 2. Combine and filter for positive binding pairs
# ---------------------------------------------------------------------------


def build_binding_pairs(
    df_davis: pd.DataFrame,
    df_kiba: pd.DataFrame,
    davis_kd_threshold: float = 10000.0,
    kiba_score_threshold: float = 12.1,
) -> pd.DataFrame:
    """Combine DAVIS and KIBA into a single set of positive binding pairs.

    Parameters
    ----------
    df_davis : pd.DataFrame
        Raw DAVIS data. Y = Kd in nM (10000 = non-binder).
    df_kiba : pd.DataFrame
        Raw KIBA data. Y = KIBA score (lower = stronger binding).
    davis_kd_threshold : float
        DAVIS Kd threshold in nM. Pairs with Kd < threshold are binders.
    kiba_score_threshold : float
        KIBA score threshold. Pairs with score ≤ threshold are binders.

    Returns
    -------
    pd.DataFrame
        Combined positive binding pairs with standardised columns.
    """
    # Filter DAVIS binders
    davis_binders = df_davis[df_davis["Y"] < davis_kd_threshold].copy()
    davis_binders["source"] = "DAVIS"
    davis_binders["drug_smiles"] = davis_binders["Drug"]
    davis_binders["target_seq"] = davis_binders["Target"]
    davis_binders["drug_id"] = davis_binders["Drug_ID"].astype(str)
    davis_binders["target_id"] = davis_binders["Target_ID"]
    print(f"  DAVIS binders (Kd < {davis_kd_threshold} nM): {len(davis_binders):,}")

    # Filter KIBA actives
    kiba_actives = df_kiba[df_kiba["Y"] <= kiba_score_threshold].copy()
    kiba_actives["source"] = "KIBA"
    kiba_actives["drug_smiles"] = kiba_actives["Drug"]
    kiba_actives["target_seq"] = kiba_actives["Target"]
    kiba_actives["drug_id"] = kiba_actives["Drug_ID"].astype(str)
    kiba_actives["target_id"] = kiba_actives["Target_ID"]
    print(f"  KIBA actives (score ≤ {kiba_score_threshold}): {len(kiba_actives):,}")

    # Combine
    cols = ["drug_smiles", "drug_id", "target_seq", "target_id", "source"]
    combined = pd.concat([davis_binders[cols], kiba_actives[cols]], ignore_index=True)

    # Deduplicate on (drug_smiles, target_seq) — keep first occurrence
    before = len(combined)
    combined = combined.drop_duplicates(subset=["drug_smiles", "target_seq"], keep="first")
    print(f"  Combined: {before:,} → {len(combined):,} after deduplication")
    print(f"  Unique drugs: {combined.drug_smiles.nunique():,}")
    print(f"  Unique targets: {combined.target_seq.nunique():,}")

    return combined.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Build metadata with cluster IDs for OOD splitting
# ---------------------------------------------------------------------------


def build_metadata(pairs: pd.DataFrame) -> pd.DataFrame:
    """Build metadata DataFrame in CALM format.

    Assigns unique IDs for drugs (ag) and targets (ab), and creates
    a cluster_id based on target_id for OOD splitting.

    Note on naming convention: CALM uses "ag" for agent and "ab" for the
    other molecule (originally antibody/antigen). For dtSFM:
      ag = drug (agent molecule)
      ab = protein target

    Parameters
    ----------
    pairs : pd.DataFrame
        Combined binding pairs.

    Returns
    -------
    pd.DataFrame
        Metadata ready for CALM pipeline.
    """
    # Map unique SMILES to integer IDs
    unique_drugs = pairs["drug_smiles"].unique()
    drug_to_id = {s: i for i, s in enumerate(unique_drugs)}

    # Map unique target sequences to integer IDs
    unique_targets = pairs["target_seq"].unique()
    target_to_id = {s: i for i, s in enumerate(unique_targets)}

    # Map target_id (kinase name / UniProt ID) to cluster_id
    # This groups all pairs with the same kinase for OOD splitting
    unique_target_ids = pairs["target_id"].unique()
    target_id_to_cluster = {tid: i for i, tid in enumerate(unique_target_ids)}

    df = pairs.copy()
    df["pair_index"] = range(len(df))
    df["ag_id"] = df["drug_smiles"].map(drug_to_id)
    df["ab_id"] = df["target_seq"].map(target_to_id)
    df["cluster_id"] = df["target_id"].map(target_id_to_cluster)
    # Unique hash for split indexing (matches CALM convention)
    df["Unique_ag_vh_vl_hash"] = (
        df["drug_id"] + "_" + df["target_id"] + "_" + df["pair_index"].astype(str)
    )
    # Keep protein_seq column for MMseqs2 clustering
    df["protein_seq"] = df["target_seq"]

    print(f"  Metadata: {len(df):,} pairs, {df.ag_id.nunique():,} drugs, "
          f"{df.ab_id.nunique():,} targets, {df.cluster_id.nunique():,} clusters")

    return df


# ---------------------------------------------------------------------------
# 4. Embed drugs with MoLFormer-XL
# ---------------------------------------------------------------------------


def embed_drugs_molformer(
    smiles_list: list[str],
    model_name: str = "ibm/MoLFormer-XL-both-10pct",
    device: str = "cpu",
    batch_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Embed drug SMILES strings using MoLFormer-XL.

    MoLFormer is a molecular language model pre-trained on ~110M molecules.
    It reads SMILES strings (text notation for chemical structures) the same
    way ESM-2 reads protein sequences.

    Parameters
    ----------
    smiles_list : list[str]
        SMILES strings for each drug.
    model_name : str
        HuggingFace model name.
    device : str
        Device for inference.
    batch_size : int
        Batch size for inference.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        embeddings: (N, L_max, 768), masks: (N, L_max) boolean
    """
    from transformers import AutoModel, AutoTokenizer

    print(f"  Loading MoLFormer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name, deterministic_eval=True, trust_remote_code=True
    )
    model = model.to(device)
    model.eval()

    all_embeddings = []
    all_lengths = []

    with torch.no_grad():
        for start in range(0, len(smiles_list), batch_size):
            end = min(start + batch_size, len(smiles_list))
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
            # last_hidden_state: (B, L_tokens, 768)
            if isinstance(outputs, tuple):
                hidden_states = outputs[0]
            else:
                hidden_states = outputs.last_hidden_state

            attention_mask = inputs["attention_mask"]

            for j in range(hidden_states.shape[0]):
                valid_len = attention_mask[j].sum().item()
                emb = hidden_states[j, :valid_len, :].cpu()
                all_embeddings.append(emb)
                all_lengths.append(valid_len)

            if end % 200 == 0 or end == len(smiles_list):
                print(f"  MoLFormer: embedded {end}/{len(smiles_list)} drugs")

    # Pad to uniform length
    max_len = max(all_lengths)
    embed_dim = all_embeddings[0].shape[1]  # 768
    N = len(all_embeddings)

    padded = torch.zeros(N, max_len, embed_dim)
    masks = torch.zeros(N, max_len, dtype=torch.bool)

    for i, emb in enumerate(all_embeddings):
        L = emb.shape[0]
        padded[i, :L, :] = emb
        masks[i, :L] = True

    print(f"  MoLFormer embeddings: {padded.shape}, masks: {masks.shape}")
    return padded, masks


# ---------------------------------------------------------------------------
# 5. Embed proteins with ESM-2 (reuse from jaspar.py)
# ---------------------------------------------------------------------------


def embed_proteins_esm2(
    sequences: list[str],
    model_name: str = "esm2_t33_650M_UR50D",
    device: str = "cpu",
    batch_size: int = 4,
    max_seq_len: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Embed protein sequences using ESM-2.

    Same as jaspar.py but with higher max_seq_len default (kinase proteins
    are larger than DNA-binding domains, typically 300-1000 residues).

    Parameters
    ----------
    sequences : list[str]
        Protein sequences.
    model_name : str
        ESM-2 model name.
    device : str
        Device for inference.
    batch_size : int
        Batch size.
    max_seq_len : int
        Maximum sequence length (truncate longer sequences).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        embeddings: (N, L_max, 1280), masks: (N, L_max) boolean
    """
    import esm

    sequences_trunc = [seq[:max_seq_len] for seq in sequences]
    n_truncated = sum(1 for s in sequences if len(s) > max_seq_len)
    if n_truncated > 0:
        print(f"  Truncated {n_truncated}/{len(sequences)} sequences to {max_seq_len} residues")

    print(f"  Loading ESM-2: {model_name}...")
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    all_embeddings = []
    all_lengths = []

    with torch.no_grad():
        for start in range(0, len(sequences_trunc), batch_size):
            end = min(start + batch_size, len(sequences_trunc))
            batch_seqs = [
                (f"seq_{i}", seq)
                for i, seq in enumerate(sequences_trunc[start:end])
            ]
            _, _, batch_tokens = batch_converter(batch_seqs)
            batch_tokens = batch_tokens.to(device)

            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_reps = results["representations"][33]

            for j in range(token_reps.shape[0]):
                seq_len = len(sequences_trunc[start + j])
                emb = token_reps[j, 1 : seq_len + 1, :].cpu()
                all_embeddings.append(emb)
                all_lengths.append(seq_len)

            if end % 50 == 0 or end == len(sequences):
                print(f"  ESM-2: embedded {end}/{len(sequences)} proteins")

    max_len = max(all_lengths)
    embed_dim = all_embeddings[0].shape[1]  # 1280
    N = len(all_embeddings)

    padded = torch.zeros(N, max_len, embed_dim)
    masks = torch.zeros(N, max_len, dtype=torch.bool)

    for i, emb in enumerate(all_embeddings):
        L = emb.shape[0]
        padded[i, :L, :] = emb
        masks[i, :L] = True

    print(f"  ESM-2 embeddings: {padded.shape}, masks: {masks.shape}")
    return padded, masks


# ---------------------------------------------------------------------------
# 6. Build train/val/test splits
# ---------------------------------------------------------------------------


def build_splits(
    df: pd.DataFrame,
    output_dir: Path,
    n_folds: int = 5,
    seed: int = 42,
) -> None:
    """Create cluster-based CV splits for OOD evaluation.

    Clusters are kinase identities (target_id). All pairs involving the
    same kinase go into the same fold, ensuring OOD evaluation tests
    generalization to unseen kinase families.

    Parameters
    ----------
    df : pd.DataFrame
        Metadata with cluster_id column.
    output_dir : Path
        Directory to save split JSON files.
    n_folds : int
        Number of CV folds.
    seed : int
        Random seed.
    """
    import random

    output_dir.mkdir(parents=True, exist_ok=True)

    cluster_ids = df["cluster_id"].unique().tolist()
    random.seed(seed)
    random.shuffle(cluster_ids)

    n_test = max(1, len(cluster_ids) // n_folds)
    n_val = max(1, len(cluster_ids) // n_folds)

    for outer_fold in range(n_folds):
        test_start = outer_fold * n_test
        test_end = min(test_start + n_test, len(cluster_ids))
        test_clusters = set(cluster_ids[test_start:test_end])
        trainval_clusters = [c for c in cluster_ids if c not in test_clusters]

        random.shuffle(trainval_clusters)

        for inner_fold in range(n_folds):
            val_start = inner_fold * n_val
            val_end = min(val_start + n_val, len(trainval_clusters))
            val_clusters = set(trainval_clusters[val_start:val_end])
            train_clusters = set(trainval_clusters) - val_clusters

            split_hashes = {
                "train": df[df["cluster_id"].isin(train_clusters)][
                    "Unique_ag_vh_vl_hash"
                ].tolist(),
                "val": df[df["cluster_id"].isin(val_clusters)][
                    "Unique_ag_vh_vl_hash"
                ].tolist(),
                "test": df[df["cluster_id"].isin(test_clusters)][
                    "Unique_ag_vh_vl_hash"
                ].tolist(),
            }

            filename = (
                output_dir
                / f"split_hash_ids_outerfold_{outer_fold}_innerfold_{inner_fold}.json"
            )
            with open(filename, "w") as f:
                json.dump(split_hashes, f)

    print(f"  Saved {n_folds * n_folds} split files to {output_dir}")


# ---------------------------------------------------------------------------
# 7. Main pipeline
# ---------------------------------------------------------------------------


def run_download(output_dir: str) -> None:
    """Download and combine datasets (no GPU needed).

    Run this on your Mac or on the Euler login node.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Step 1: Downloading DAVIS + KIBA...")
    df_davis, df_kiba = download_datasets(out)

    print("Step 2: Filtering for positive binding pairs...")
    pairs = build_binding_pairs(df_davis, df_kiba)

    print("Step 3: Building metadata...")
    df = build_metadata(pairs)
    df.to_csv(out / "metadata.csv", index=False)

    print("Step 4: Building initial splits (by target_id)...")
    build_splits(df, out / "splits")

    print(f"\nDownload complete! Saved to {out}")
    print(f"  metadata.csv:  {len(df):,} binding pairs")
    print(f"  davis_raw.csv: raw DAVIS data")
    print(f"  kiba_raw.csv:  raw KIBA data")
    print(f"  splits/:       CV split files")
    print(f"\nNext step: run 'embed' subcommand on Euler GPU node")


def run_embed(
    output_dir: str,
    device: str = "cpu",
    molformer_batch_size: int = 32,
    esm_batch_size: int = 4,
) -> None:
    """Embed drugs and proteins (requires GPU).

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

    # --- Embed unique drugs with MoLFormer ---
    unique_drugs = df["drug_smiles"].drop_duplicates().tolist()
    drug_to_idx = {s: i for i, s in enumerate(unique_drugs)}
    ag_indices = torch.tensor(
        [drug_to_idx[s] for s in df["drug_smiles"]], dtype=torch.long
    )

    print(f"\nStep 1: Embedding {len(unique_drugs):,} unique drugs with MoLFormer "
          f"(from {len(df):,} total pairs)...")
    ag_embed, ag_mask = embed_drugs_molformer(
        unique_drugs, device=device, batch_size=molformer_batch_size
    )
    torch.save(ag_embed, out / "ag_embed.pt")
    torch.save(ag_mask, out / "ag_mask.pt")
    torch.save(ag_indices, out / "ag_indices.pt")
    print(f"  Saved ag_embed.pt {ag_embed.shape}, ag_mask.pt, ag_indices.pt")

    # --- Embed unique proteins with ESM-2 ---
    unique_targets = df["target_seq"].drop_duplicates().tolist()
    target_to_idx = {s: i for i, s in enumerate(unique_targets)}
    ab_indices = torch.tensor(
        [target_to_idx[s] for s in df["target_seq"]], dtype=torch.long
    )

    print(f"\nStep 2: Embedding {len(unique_targets):,} unique proteins with ESM-2 "
          f"(from {len(df):,} total pairs)...")
    ab_embed, ab_mask = embed_proteins_esm2(
        unique_targets, device=device, batch_size=esm_batch_size
    )
    torch.save(ab_embed, out / "ab_embed.pt")
    torch.save(ab_mask, out / "ab_mask.pt")
    torch.save(ab_indices, out / "ab_indices.pt")
    print(f"  Saved ab_embed.pt {ab_embed.shape}, ab_mask.pt, ab_indices.pt")

    print(f"\nEmbedding complete! All tensors saved to {out}")
    print(f"  ag_embed.pt:     Drug embeddings (MoLFormer, 768-dim)")
    print(f"  ab_embed.pt:     Protein embeddings (ESM-2, 1280-dim)")
    print(f"  ag_indices.pt:   Maps each row → unique drug index")
    print(f"  ab_indices.pt:   Maps each row → unique protein index")
    print(f"  ag_mask.pt:      Drug padding masks")
    print(f"  ab_mask.pt:      Protein padding masks")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess DAVIS + KIBA data for dtSFM training"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- download subcommand ---
    dl = subparsers.add_parser(
        "download",
        help="Download DAVIS + KIBA and build metadata (no GPU needed)",
    )
    dl.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save outputs",
    )

    # --- embed subcommand ---
    em = subparsers.add_parser(
        "embed",
        help="Embed drugs (MoLFormer) and proteins (ESM-2) — requires GPU",
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
