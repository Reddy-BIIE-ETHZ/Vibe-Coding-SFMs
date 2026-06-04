"""
Preprocessing pipeline for tSFM: Transcription Factor–DNA binding.

Downloads JASPAR CORE vertebrate TF profiles, fetches TF protein sequences
from UniProt, computes consensus DNA sequences from PFMs, embeds both via
ESM-2 (protein) and DNABERT-2 (DNA), and saves CALM-ready tensors.

Usage:
    python -m calm.preprocess.jaspar \
        --output_dir /path/to/output \
        --max_profiles 0           # 0 = all profiles
        --device cuda              # or cpu
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
    torch = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 1. JASPAR API: fetch TF profiles
# ---------------------------------------------------------------------------

JASPAR_API = "https://jaspar.elixir.no/api/v1"


def fetch_jaspar_profiles(
    tax_group: str = "vertebrates",
    collection: str = "CORE",
    max_profiles: int = 0,
) -> list[dict]:
    """Fetch all TF matrix profiles from JASPAR REST API.

    Parameters
    ----------
    tax_group : str
        Taxonomic group (e.g., "vertebrates").
    collection : str
        JASPAR collection (e.g., "CORE").
    max_profiles : int
        Maximum number of profiles to fetch (0 = all).

    Returns
    -------
    list[dict]
        List of profile dictionaries with PFM, metadata, UniProt IDs.
    """
    profiles = []
    page = 1
    page_size = 100

    while True:
        url = (
            f"{JASPAR_API}/matrix/?format=json"
            f"&collection={collection}"
            f"&tax_group={tax_group}"
            f"&page_size={page_size}"
            f"&page={page}"
        )
        print(f"  Fetching JASPAR page {page}...")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        for entry in data["results"]:
            # Fetch full profile (the listing doesn't include PFM)
            detail_url = entry["url"] + "?format=json"
            detail = requests.get(detail_url, timeout=30)
            detail.raise_for_status()
            profiles.append(detail.json())

            if max_profiles > 0 and len(profiles) >= max_profiles:
                break

        if max_profiles > 0 and len(profiles) >= max_profiles:
            break
        if data["next"] is None:
            break
        page += 1
        time.sleep(0.2)  # be polite to the API

    print(f"  Fetched {len(profiles)} JASPAR profiles.")
    return profiles


# ---------------------------------------------------------------------------
# 2. Consensus DNA sequence from PFM
# ---------------------------------------------------------------------------


def pfm_to_consensus(pfm: dict[str, list[float]]) -> str:
    """Convert a Position Frequency Matrix to a consensus DNA sequence.

    At each position, the nucleotide with the highest frequency is chosen.

    Parameters
    ----------
    pfm : dict
        PFM with keys "A", "C", "G", "T" and lists of counts.

    Returns
    -------
    str
        Consensus DNA sequence.
    """
    length = len(pfm["A"])
    consensus = []
    for i in range(length):
        counts = {nt: pfm[nt][i] for nt in "ACGT"}
        consensus.append(max(counts, key=counts.get))
    return "".join(consensus)


def sample_sequences_from_pfm(
    pfm: dict[str, list[float]],
    n_samples: int,
    seed: int | None = None,
) -> list[str]:
    """Sample DNA sequences from a Position Frequency Matrix.

    At each position, nucleotides are drawn according to their frequencies
    in the PFM. This generates realistic binding site variants that reflect
    the TF's actual specificity landscape.

    Parameters
    ----------
    pfm : dict
        PFM with keys "A", "C", "G", "T" and lists of counts.
    n_samples : int
        Number of sequences to sample.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    list[str]
        List of sampled DNA sequences.
    """
    import random as rng

    if seed is not None:
        rng.seed(seed)

    length = len(pfm["A"])
    nucleotides = list("ACGT")
    sequences = []

    for _ in range(n_samples):
        seq = []
        for i in range(length):
            weights = [pfm[nt][i] for nt in nucleotides]
            total = sum(weights)
            # Normalize to probabilities
            probs = [w / total for w in weights]
            # Weighted random choice
            chosen = rng.choices(nucleotides, weights=probs, k=1)[0]
            seq.append(chosen)
        sequences.append("".join(seq))

    return sequences


# ---------------------------------------------------------------------------
# 3. Fetch protein sequences from UniProt
# ---------------------------------------------------------------------------


def fetch_uniprot_sequence(uniprot_id: str) -> str | None:
    """Fetch protein sequence from UniProt REST API.

    Parameters
    ----------
    uniprot_id : str
        UniProt accession ID.

    Returns
    -------
    str or None
        Protein sequence, or None if not found.
    """
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        lines = resp.text.strip().split("\n")
        # Skip header line starting with >
        seq = "".join(line for line in lines if not line.startswith(">"))
        return seq
    except Exception as e:
        print(f"  Warning: Failed to fetch UniProt {uniprot_id}: {e}")
        return None


def fetch_protein_sequences(profiles: list[dict]) -> list[dict]:
    """For each JASPAR profile, fetch the TF protein sequence from UniProt.

    Profiles without a UniProt ID or with failed lookups are skipped.

    Parameters
    ----------
    profiles : list[dict]
        JASPAR profile dictionaries.

    Returns
    -------
    list[dict]
        Filtered list with added 'protein_seq' and 'uniprot_id' fields.
    """
    results = []
    for i, p in enumerate(profiles):
        uniprot_ids = p.get("uniprot_ids", [])
        if not uniprot_ids:
            print(f"  Skipping {p['matrix_id']} ({p['name']}): no UniProt ID")
            continue

        # Use the first UniProt ID
        uid = uniprot_ids[0]
        seq = fetch_uniprot_sequence(uid)
        if seq is None or len(seq) == 0:
            print(f"  Skipping {p['matrix_id']} ({p['name']}): UniProt {uid} not found")
            continue

        p["protein_seq"] = seq
        p["uniprot_id"] = uid
        results.append(p)

        if (i + 1) % 50 == 0:
            print(f"  Fetched {len(results)} protein sequences ({i+1}/{len(profiles)} checked)...")
            time.sleep(0.5)  # rate limiting

    print(f"  Got protein sequences for {len(results)}/{len(profiles)} profiles.")
    return results


# ---------------------------------------------------------------------------
# 4. Embed proteins with ESM-2
# ---------------------------------------------------------------------------


def embed_proteins_esm2(
    sequences: list[str],
    model_name: str = "esm2_t33_650M_UR50D",
    device: str = "cpu",
    batch_size: int = 4,
    max_seq_len: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Embed protein sequences using ESM-2.

    Parameters
    ----------
    sequences : list[str]
        Protein sequences (amino acid strings).
    model_name : str
        ESM-2 model name.
    device : str
        Device to run inference on.
    batch_size : int
        Batch size for inference.
    max_seq_len : int
        Maximum sequence length. Longer sequences are truncated.
        Default 512 covers most DNA-binding domains while keeping
        the output tensor manageable.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        embeddings: (N, L_max, 1280), masks: (N, L_max) boolean
    """
    import esm

    # Truncate sequences to max_seq_len
    sequences_trunc = [seq[:max_seq_len] for seq in sequences]
    n_truncated = sum(1 for s in sequences if len(s) > max_seq_len)
    if n_truncated > 0:
        print(f"  Truncated {n_truncated}/{len(sequences)} sequences to {max_seq_len} residues")

    print(f"  Loading ESM-2 model: {model_name}...")
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    all_embeddings = []
    all_lengths = []

    with torch.no_grad():
        for start in range(0, len(sequences_trunc), batch_size):
            end = min(start + batch_size, len(sequences_trunc))
            batch_seqs = [(f"seq_{i}", seq) for i, seq in enumerate(sequences_trunc[start:end])]
            _, _, batch_tokens = batch_converter(batch_seqs)
            batch_tokens = batch_tokens.to(device)

            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            # Shape: (B, L+2, 1280) — includes BOS and EOS tokens
            token_reps = results["representations"][33]

            # Remove BOS (index 0) and EOS (last token) — keep residue embeddings only
            for j in range(token_reps.shape[0]):
                seq_len = len(sequences_trunc[start + j])
                # Residue embeddings are at positions 1..seq_len (0 is BOS, seq_len+1 is EOS)
                emb = token_reps[j, 1 : seq_len + 1, :].cpu()  # (seq_len, 1280)
                all_embeddings.append(emb)
                all_lengths.append(seq_len)

            if (end) % 100 == 0 or end == len(sequences):
                print(f"  ESM-2: embedded {end}/{len(sequences)} sequences")

    # Pad to uniform length
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
# 5. Embed DNA sequences with DNABERT-2
# ---------------------------------------------------------------------------


def embed_dna_dnabert2(
    sequences: list[str],
    model_name: str = "zhihan1996/DNABERT-2-117M",
    device: str = "cpu",
    batch_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Embed DNA sequences using DNABERT-2.

    Parameters
    ----------
    sequences : list[str]
        DNA sequences (nucleotide strings, e.g. "ACGTACGT").
    model_name : str
        Hugging Face model name for DNABERT-2.
    device : str
        Device to run inference on.
    batch_size : int
        Batch size for inference.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        embeddings: (N, L_max, 768), masks: (N, L_max) boolean
    """
    from transformers import AutoModel, AutoTokenizer
    from transformers.models.bert.configuration_bert import BertConfig

    print(f"  Loading DNABERT-2 model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = BertConfig.from_pretrained(model_name)
    # Disable flash attention to avoid triton version incompatibility
    config.use_flash_attn = False
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, config=config)
    model = model.to(device)
    model.eval()

    all_embeddings = []
    all_lengths = []

    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            end = min(start + batch_size, len(sequences))
            batch_seqs = sequences[start:end]

            inputs = tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            # hidden_states: (B, L_tokens, 768)
            # Handle both tuple returns (older transformers) and named returns
            if isinstance(outputs, tuple):
                hidden_states = outputs[0]
            else:
                hidden_states = outputs.last_hidden_state

            attention_mask = inputs["attention_mask"]

            for j in range(hidden_states.shape[0]):
                # Get the valid (non-padding) token embeddings
                valid_len = attention_mask[j].sum().item()
                # Remove [CLS] and [SEP] tokens if present
                # DNABERT-2 uses BPE tokenization — tokens correspond to sub-sequences
                # We keep all non-padding token embeddings as the sequence representation
                emb = hidden_states[j, :valid_len, :].cpu()  # (valid_len, 768)
                all_embeddings.append(emb)
                all_lengths.append(valid_len)

            if end % 100 == 0 or end == len(sequences):
                print(f"  DNABERT-2: embedded {end}/{len(sequences)} sequences")

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

    print(f"  DNABERT-2 embeddings: {padded.shape}, masks: {masks.shape}")
    return padded, masks


# ---------------------------------------------------------------------------
# 6. Build metadata and splits
# ---------------------------------------------------------------------------


def build_metadata(
    profiles: list[dict],
    samples_per_pwm: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """Build metadata DataFrame from processed JASPAR profiles.

    Parameters
    ----------
    profiles : list[dict]
        JASPAR profiles with protein_seq and consensus_dna fields.
    samples_per_pwm : int
        Number of DNA sequences to sample per PWM. If 1, uses the
        consensus sequence only. If > 1, samples from the PWM
        frequency distribution to generate diverse binding sites.
    seed : int
        Random seed for PWM sampling.

    Returns
    -------
    pd.DataFrame
        Metadata with columns needed by CALM training pipeline.
    """
    records = []
    # Map unique TF names to integer IDs
    tf_names = list({p["name"] for p in profiles})
    tf_name_to_id = {name: i for i, name in enumerate(tf_names)}

    # Map TF families to cluster IDs for OOD splitting
    families = list({p.get("family", ["unknown"])[0] if p.get("family") else "unknown" for p in profiles})
    family_to_cluster = {fam: i for i, fam in enumerate(families)}

    # Track all unique DNA sequences for ab_id assignment
    all_dna_seqs: dict[str, int] = {}
    dna_id_counter = 0

    pair_index = 0
    for p in profiles:
        family = p.get("family", ["unknown"])[0] if p.get("family") else "unknown"
        tf_class = p.get("class", ["unknown"])[0] if p.get("class") else "unknown"
        species_name = p["species"][0]["name"] if p.get("species") else "unknown"

        # Generate DNA sequences: consensus + sampled variants
        if samples_per_pwm <= 1:
            dna_sequences = [p["consensus_dna"]]
        else:
            # Always include consensus, then sample (n-1) variants
            sampled = sample_sequences_from_pfm(
                p["pfm"],
                n_samples=samples_per_pwm - 1,
                seed=seed + pair_index,
            )
            dna_sequences = [p["consensus_dna"]] + sampled

        for dna_seq in dna_sequences:
            if dna_seq not in all_dna_seqs:
                all_dna_seqs[dna_seq] = dna_id_counter
                dna_id_counter += 1

            records.append({
                "pair_index": pair_index,
                "matrix_id": p["matrix_id"],
                "tf_name": p["name"],
                "uniprot_id": p["uniprot_id"],
                "protein_seq": p["protein_seq"],
                "consensus_dna": dna_seq,
                "tf_family": family,
                "tf_class": tf_class,
                "species": species_name,
                "ag_id": tf_name_to_id[p["name"]],
                "ab_id": all_dna_seqs[dna_seq],
                "cluster_id": family_to_cluster[family],
                "Unique_ag_vh_vl_hash": f"{p['matrix_id']}_{p['uniprot_id']}_{pair_index}",
            })
            pair_index += 1

    df = pd.DataFrame(records)
    print(f"  Metadata: {len(df)} pairs ({samples_per_pwm} per PWM), "
          f"{df['ag_id'].nunique()} unique TFs, "
          f"{df['cluster_id'].nunique()} clusters (TF families)")
    return df


def build_splits(
    df: pd.DataFrame,
    output_dir: Path,
    n_folds: int = 5,
    seed: int = 42,
) -> None:
    """Create train/val/test split index files based on TF family clusters.

    Uses the same cluster-based OOD splitting strategy as CALM:
    clusters are assigned to folds, ensuring that all members of a
    TF family are in the same split.

    Parameters
    ----------
    df : pd.DataFrame
        Metadata DataFrame with cluster_id column.
    output_dir : Path
        Directory to save split index JSON files.
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
        # Test clusters
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
                "train": df[df["cluster_id"].isin(train_clusters)]["Unique_ag_vh_vl_hash"].tolist(),
                "val": df[df["cluster_id"].isin(val_clusters)]["Unique_ag_vh_vl_hash"].tolist(),
                "test": df[df["cluster_id"].isin(test_clusters)]["Unique_ag_vh_vl_hash"].tolist(),
            }

            filename = output_dir / f"split_hash_ids_outerfold_{outer_fold}_innerfold_{inner_fold}.json"
            with open(filename, "w") as f:
                json.dump(split_hashes, f)

    print(f"  Saved {n_folds * n_folds} split files to {output_dir}")


# ---------------------------------------------------------------------------
# 7. Main pipeline
# ---------------------------------------------------------------------------


def run_preprocessing(
    output_dir: str,
    max_profiles: int = 0,
    device: str = "cpu",
    esm_batch_size: int = 4,
    dna_batch_size: int = 16,
    skip_embedding: bool = False,
    samples_per_pwm: int = 1,
) -> None:
    """Run the full JASPAR→tSFM preprocessing pipeline.

    Parameters
    ----------
    output_dir : str
        Directory to save all output files.
    max_profiles : int
        Maximum profiles to process (0 = all).
    device : str
        Device for model inference.
    esm_batch_size : int
        Batch size for ESM-2 inference.
    dna_batch_size : int
        Batch size for DNABERT-2 inference.
    skip_embedding : bool
        If True, skip embedding step (use if tensors already exist).
    samples_per_pwm : int
        Number of DNA sequences to sample per PWM (1 = consensus only).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Fetch JASPAR profiles ---
    profiles_cache = out / "jaspar_profiles.json"
    if profiles_cache.exists():
        print("Step 1: Loading cached JASPAR profiles...")
        with open(profiles_cache) as f:
            profiles = json.load(f)
    else:
        print("Step 1: Fetching JASPAR profiles...")
        profiles = fetch_jaspar_profiles(max_profiles=max_profiles)
        with open(profiles_cache, "w") as f:
            json.dump(profiles, f)

    # --- Step 2: Fetch protein sequences ---
    profiles_with_seq_cache = out / "profiles_with_sequences.json"
    if profiles_with_seq_cache.exists():
        print("Step 2: Loading cached protein sequences...")
        with open(profiles_with_seq_cache) as f:
            profiles = json.load(f)
    else:
        print("Step 2: Fetching TF protein sequences from UniProt...")
        profiles = fetch_protein_sequences(profiles)
        with open(profiles_with_seq_cache, "w") as f:
            json.dump(profiles, f)

    # --- Step 3: Compute consensus DNA sequences ---
    print("Step 3: Computing consensus DNA sequences from PFMs...")
    for p in profiles:
        p["consensus_dna"] = pfm_to_consensus(p["pfm"])

    # --- Step 4: Build metadata ---
    print(f"Step 4: Building metadata ({samples_per_pwm} samples per PWM)...")
    df = build_metadata(profiles, samples_per_pwm=samples_per_pwm)
    df.to_csv(out / "metadata.csv", index=False)

    # --- Step 5: Embed UNIQUE sequences + save index mappings ---
    # With 10 samples per PWM, many rows share the same protein or DNA sequence.
    # Embedding only unique sequences saves memory and time:
    #   20,540 proteins → ~1,060 unique → tensor drops from ~54 GB to ~2.8 GB
    #   20,540 DNA seqs  → ~15,400 unique (small anyway)
    if not skip_embedding:
        # --- 5a: Unique proteins ---
        unique_proteins = df["protein_seq"].drop_duplicates().tolist()
        prot_to_idx = {seq: i for i, seq in enumerate(unique_proteins)}
        ag_indices = torch.tensor([prot_to_idx[s] for s in df["protein_seq"]], dtype=torch.long)

        print(f"Step 5a: Embedding {len(unique_proteins)} unique TF proteins with ESM-2 "
              f"(from {len(df)} total rows)...")
        ag_embed, ag_mask = embed_proteins_esm2(
            unique_proteins, device=device, batch_size=esm_batch_size
        )
        torch.save(ag_embed, out / "ag_embed.pt")
        torch.save(ag_mask, out / "ag_mask.pt")
        torch.save(ag_indices, out / "ag_indices.pt")
        print(f"  Saved ag_embed.pt {ag_embed.shape}, ag_mask.pt, ag_indices.pt ({len(ag_indices)} mappings)")

        # --- 5b: Unique DNA sequences ---
        unique_dna = df["consensus_dna"].drop_duplicates().tolist()
        dna_to_idx = {seq: i for i, seq in enumerate(unique_dna)}
        ab_indices = torch.tensor([dna_to_idx[s] for s in df["consensus_dna"]], dtype=torch.long)

        print(f"Step 5b: Embedding {len(unique_dna)} unique DNA sequences with DNABERT-2 "
              f"(from {len(df)} total rows)...")
        ab_embed, ab_mask = embed_dna_dnabert2(
            unique_dna, device=device, batch_size=dna_batch_size
        )
        torch.save(ab_embed, out / "ab_embed.pt")
        torch.save(ab_mask, out / "ab_mask.pt")
        torch.save(ab_indices, out / "ab_indices.pt")
        print(f"  Saved ab_embed.pt {ab_embed.shape}, ab_mask.pt, ab_indices.pt ({len(ab_indices)} mappings)")
    else:
        print("Step 5: Skipping embedding (--skip_embedding set)")

    # --- Step 6: Build train/val/test splits ---
    print("Step 6: Building train/val/test splits...")
    split_dir = out / "splits"
    build_splits(df, split_dir)

    print(f"\nDone! All outputs saved to {out}")
    print(f"  metadata.csv:    {len(df)} TF-DNA pairs")
    print(f"  ag_embed.pt:     Unique TF protein embeddings (ESM-2, 1280-dim)")
    print(f"  ab_embed.pt:     Unique DNA sequence embeddings (DNABERT-2, 768-dim)")
    print(f"  ag_indices.pt:   Maps each row → unique protein index")
    print(f"  ab_indices.pt:   Maps each row → unique DNA index")
    print(f"  ag_mask.pt:      TF protein padding masks")
    print(f"  ab_mask.pt:      DNA sequence padding masks")
    print(f"  splits/:         CV split index files")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess JASPAR data for tSFM training"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save preprocessed outputs",
    )
    parser.add_argument(
        "--max_profiles",
        type=int,
        default=0,
        help="Max JASPAR profiles to fetch (0 = all, use small number for testing)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for embedding models (cpu or cuda)",
    )
    parser.add_argument(
        "--esm_batch_size",
        type=int,
        default=4,
        help="Batch size for ESM-2 inference",
    )
    parser.add_argument(
        "--dna_batch_size",
        type=int,
        default=16,
        help="Batch size for DNABERT-2 inference",
    )
    parser.add_argument(
        "--skip_embedding",
        action="store_true",
        help="Skip embedding step (use if tensors already computed)",
    )
    parser.add_argument(
        "--samples_per_pwm",
        type=int,
        default=1,
        help="DNA sequences to sample per PWM (1 = consensus only, e.g. 10 or 20 for diversity)",
    )
    args = parser.parse_args()

    run_preprocessing(
        output_dir=args.output_dir,
        max_profiles=args.max_profiles,
        device=args.device,
        esm_batch_size=args.esm_batch_size,
        dna_batch_size=args.dna_batch_size,
        skip_embedding=args.skip_embedding,
        samples_per_pwm=args.samples_per_pwm,
    )
