"""
Preprocessing pipeline for miR-SFM: miRNA–mRNA target binding specificity.

Reads pre-extracted (miRNA_seq, target_site_seq) pairs from Step 1,
embeds both with DNABERT-2, and saves CALM-ready tensors with
index-based lookup (same pattern as data_tsfm.py).

Data comes from:
  - miRTarBase: experimentally validated miRNA-target interactions
  - ENCORI: CLIP-seq binding site coordinates
  - hg38: target site sequences extracted at binding coordinates

Both agent (miRNA, ~22nt) and target (binding site, 30nt) are short
nucleotide sequences already U→T converted for DNABERT-2.

Usage:
    python -m calm.preprocess.mirsfm \\
        --training_file data/mirsfm/mirsfm_training.tsv \\
        --output_dir /path/to/output \\
        --device cuda

    # Skip embedding if tensors already exist:
    python -m calm.preprocess.mirsfm \\
        --training_file data/mirsfm/mirsfm_training.tsv \\
        --output_dir /path/to/output \\
        --skip_embedding
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 1. Load and prepare data
# ---------------------------------------------------------------------------


def load_data(
    training_file: str,
    max_pairs: int = 0,
) -> pd.DataFrame:
    """Load pre-extracted miRNA-target pairs from Step 1 TSV.

    Parameters
    ----------
    training_file : str
        Path to mirsfm_training.tsv from extract_sequences.py.
    max_pairs : int
        Maximum pairs to keep (0 = all). Useful for smoke tests.

    Returns
    -------
    pd.DataFrame
        DataFrame with mirna_name, mirna_seq, gene_name, target_seq,
        is_canonical, chrom, start, end, strand, clip_exp_num.
    """
    print(f"  Loading training data from {training_file}...")
    df = pd.read_csv(training_file, sep="\t")
    print(f"  Loaded {len(df):,} pairs")
    print(f"  Unique miRNAs: {df['mirna_name'].nunique()}")
    print(f"  Unique target genes: {df['gene_name'].nunique()}")
    print(f"  Canonical: {df['is_canonical'].sum():,}, "
          f"Non-canonical: {(~df['is_canonical'].astype(bool)).sum():,}")

    if max_pairs > 0:
        df = df.head(max_pairs)
        print(f"  Capped to {max_pairs} pairs (--max_pairs)")

    return df


# ---------------------------------------------------------------------------
# 2. Build metadata with unique IDs and index mappings
# ---------------------------------------------------------------------------


def build_metadata(df: pd.DataFrame) -> tuple[pd.DataFrame, torch.Tensor, torch.Tensor]:
    """Build metadata with unique sequence IDs and index mappings.

    For the index-based data loader (data_tsfm.py), we need:
    - Unique miRNA embeddings (N_unique_mirnas, L, 768)
    - Unique target embeddings (N_unique_targets, L, 768)
    - Index mappings: pair_i → unique_mirna_idx, pair_i → unique_target_idx

    Parameters
    ----------
    df : pd.DataFrame
        Raw pair data from load_data().

    Returns
    -------
    tuple[pd.DataFrame, torch.Tensor, torch.Tensor]
        metadata: DataFrame with ag_id, ab_id, cluster_id, Unique_ag_vh_vl_hash
        ag_indices: (N_pairs,) mapping to unique miRNA index
        ab_indices: (N_pairs,) mapping to unique target index
    """
    # Build unique sequence → index mappings
    unique_mirnas = df["mirna_seq"].unique().tolist()
    unique_targets = df["target_seq"].unique().tolist()
    mirna_to_idx = {seq: i for i, seq in enumerate(unique_mirnas)}
    target_to_idx = {seq: i for i, seq in enumerate(unique_targets)}

    print(f"  Unique miRNA sequences: {len(unique_mirnas)}")
    print(f"  Unique target sequences: {len(unique_targets)}")

    # Build index tensors
    ag_indices = torch.tensor([mirna_to_idx[s] for s in df["mirna_seq"]], dtype=torch.long)
    ab_indices = torch.tensor([target_to_idx[s] for s in df["target_seq"]], dtype=torch.long)

    # Build metadata DataFrame matching CALM conventions
    records = []
    for i, row in df.iterrows():
        records.append({
            "pair_index": i,
            "mirna_name": row["mirna_name"],
            "mirna_seq": row["mirna_seq"],
            "gene_name": row["gene_name"],
            "target_seq": row["target_seq"],
            "is_canonical": row["is_canonical"],
            "ag_id": mirna_to_idx[row["mirna_seq"]],
            "ab_id": target_to_idx[row["target_seq"]],
            "cluster_id": row["mirna_name"],  # OOD split by miRNA identity
            # CALM split system uses this column for hash-based splitting
            "Unique_ag_vh_vl_hash": f"pair_{i}",
        })

    meta = pd.DataFrame(records)
    print(f"  Metadata: {len(meta):,} pairs")
    print(f"  Clusters (miRNAs): {meta['cluster_id'].nunique()}")
    return meta, ag_indices, ab_indices, unique_mirnas, unique_targets


# ---------------------------------------------------------------------------
# 3. Build train/val/test splits
# ---------------------------------------------------------------------------


def build_splits(
    df: pd.DataFrame,
    output_dir: Path,
    n_folds: int = 5,
    seed: int = 42,
) -> None:
    """Create cluster-based OOD splits.

    All pairs for the same miRNA stay in the same split,
    so the test set contains unseen miRNAs.

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
                "train": df[df["cluster_id"].isin(train_clusters)]["Unique_ag_vh_vl_hash"].tolist(),
                "val": df[df["cluster_id"].isin(val_clusters)]["Unique_ag_vh_vl_hash"].tolist(),
                "test": df[df["cluster_id"].isin(test_clusters)]["Unique_ag_vh_vl_hash"].tolist(),
            }

            filename = output_dir / f"split_hash_ids_outerfold_{outer_fold}_innerfold_{inner_fold}.json"
            with open(filename, "w") as f:
                json.dump(split_hashes, f)

    print(f"  Saved {n_folds * n_folds} split files to {output_dir}")


# ---------------------------------------------------------------------------
# 4. Embed sequences with DNABERT-2
# ---------------------------------------------------------------------------


def embed_dna_dnabert2(
    sequences: list[str],
    model_name: str = "zhihan1996/DNABERT-2-117M",
    device: str = "cpu",
    batch_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Embed DNA sequences using DNABERT-2.

    Same function as crispr.py — reused for consistency across SFMs.

    Parameters
    ----------
    sequences : list[str]
        DNA sequences (e.g. "ACGTACGT"). Must be U→T converted already.
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

    # ── Force-disable flash attention (Triton version incompatibility) ──
    # DNABERT-2's custom bert_layers.py imports flash_attn_triton which
    # uses tl.dot(trans_b=True) unsupported by Euler's Triton version.
    #
    # Strategy: load tokenizer first (triggers cache download of custom code),
    # then patch the cached bert_layers.py on disk, then load model.
    import os
    import sys
    import shutil

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("  Tokenizer loaded. Patching cached DNABERT-2 code to disable flash attention...")

    # Find and patch ALL bert_layers.py files in the entire HuggingFace cache
    # DNABERT-2 caches under multiple paths including "DNABERT_hyphen_2_hyphen_117M"
    cache_base = os.path.expanduser("~/.cache/huggingface")
    patched = False
    for root, dirs, files in os.walk(cache_base):
        for fname in files:
            if fname == "bert_layers.py":
                fpath = os.path.join(root, fname)
                text = open(fpath).read()
                new_text = text
                # Patch 1: force use_flash_attn to False in constructor
                new_text = new_text.replace(
                    "self.use_flash_attn = getattr(config, 'use_flash_attn', False)",
                    "self.use_flash_attn = False  # PATCHED",
                )
                # Patch 2: replace all "if self.use_flash_attn:" branches
                new_text = new_text.replace(
                    "if self.use_flash_attn:",
                    "if False:  # PATCHED: flash attn disabled",
                )
                if new_text != text:
                    open(fpath, "w").write(new_text)
                    # Clear __pycache__ so Python doesn't use stale bytecode
                    pycache = os.path.join(root, "__pycache__")
                    if os.path.isdir(pycache):
                        shutil.rmtree(pycache)
                    print(f"  PATCHED: {fpath}")
                    patched = True

    if not patched:
        print("  WARNING: no bert_layers.py found to patch — flash attn may still be active")

    # Now load model — it will use the patched bert_layers.py from cache
    config = BertConfig.from_pretrained(model_name)
    config.use_flash_attn = False

    # Force reimport of patched modules
    mods_to_remove = [k for k in sys.modules if "DNABERT" in k or "bert_layers" in k]
    for k in mods_to_remove:
        del sys.modules[k]

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

            if end % 500 == 0 or end == len(sequences):
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
# 5. Main pipeline
# ---------------------------------------------------------------------------


def run_preprocessing(
    training_file: str,
    output_dir: str,
    max_pairs: int = 0,
    device: str = "cpu",
    dna_batch_size: int = 16,
    skip_embedding: bool = False,
) -> None:
    """Run the full miR-SFM preprocessing pipeline.

    Parameters
    ----------
    training_file : str
        Path to mirsfm_training.tsv from Step 1.
    output_dir : str
        Directory to save all output files.
    max_pairs : int
        Maximum pairs (0 = all). Use small number for smoke tests.
    device : str
        Device for DNABERT-2 inference.
    dna_batch_size : int
        Batch size for DNABERT-2 inference.
    skip_embedding : bool
        If True, skip embedding step (use if tensors already exist).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Load data ---
    print("Step 1: Loading miRNA-target pairs...")
    df = load_data(training_file, max_pairs=max_pairs)

    # --- Step 2: Build metadata and index mappings ---
    print("\nStep 2: Building metadata and index mappings...")
    meta, ag_indices, ab_indices, unique_mirnas, unique_targets = build_metadata(df)
    meta.to_csv(out / "metadata.csv", index=False)
    torch.save(ag_indices, out / "ag_indices.pt")
    torch.save(ab_indices, out / "ab_indices.pt")
    print(f"  Saved metadata.csv, ag_indices.pt, ab_indices.pt")

    # --- Step 3: Embed unique sequences ---
    if not skip_embedding:
        print(f"\nStep 3a: Embedding {len(unique_mirnas)} unique miRNA sequences with DNABERT-2...")
        ag_embed, ag_mask = embed_dna_dnabert2(
            unique_mirnas, device=device, batch_size=dna_batch_size,
        )
        torch.save(ag_embed, out / "ag_embed.pt")
        torch.save(ag_mask, out / "ag_mask.pt")
        print(f"  Saved ag_embed.pt {ag_embed.shape} and ag_mask.pt {ag_mask.shape}")

        print(f"\nStep 3b: Embedding {len(unique_targets):,} unique target sequences with DNABERT-2...")
        ab_embed, ab_mask = embed_dna_dnabert2(
            unique_targets, device=device, batch_size=dna_batch_size,
        )
        torch.save(ab_embed, out / "ab_embed.pt")
        torch.save(ab_mask, out / "ab_mask.pt")
        print(f"  Saved ab_embed.pt {ab_embed.shape} and ab_mask.pt {ab_mask.shape}")
    else:
        print("\nStep 3: Skipping embedding (--skip_embedding set)")

    # --- Step 4: Build train/val/test splits ---
    print("\nStep 4: Building train/val/test splits...")
    split_dir = out / "split_index" / "by_cluster_test_cv5f"
    build_splits(meta, split_dir)

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Done! All outputs saved to {out}")
    print(f"  metadata.csv:    {len(meta):,} miRNA-target pairs")
    print(f"  ag_embed.pt:     unique miRNA embeddings (DNABERT-2, 768-dim)")
    print(f"  ab_embed.pt:     unique target site embeddings (DNABERT-2, 768-dim)")
    print(f"  ag_indices.pt:   pair → unique miRNA index mapping")
    print(f"  ab_indices.pt:   pair → unique target index mapping")
    print(f"  ag_mask.pt:      miRNA padding masks")
    print(f"  ab_mask.pt:      target padding masks")
    print(f"  split_index/:    CV split index files (5×5 = 25)")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess miRNA-target data for miR-SFM training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full preprocessing on Euler GPU:
  python -m calm.preprocess.mirsfm \\
      --training_file data/mirsfm/mirsfm_training.tsv \\
      --output_dir /cluster/scratch/$USER/mirsfm/data \\
      --device cuda --dna_batch_size 16

  # Smoke test (100 pairs, CPU):
  python -m calm.preprocess.mirsfm \\
      --training_file data/mirsfm/mirsfm_training.tsv \\
      --output_dir data/mirsfm/test_output \\
      --max_pairs 100 --device cpu

  # Skip embedding (metadata + splits only):
  python -m calm.preprocess.mirsfm \\
      --training_file data/mirsfm/mirsfm_training.tsv \\
      --output_dir data/mirsfm/output \\
      --skip_embedding
""",
    )
    parser.add_argument("--training_file", required=True, help="Path to mirsfm_training.tsv")
    parser.add_argument("--output_dir", required=True, help="Directory for output files")
    parser.add_argument("--max_pairs", type=int, default=0, help="Max pairs (0=all)")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--dna_batch_size", type=int, default=16, help="DNABERT-2 batch size")
    parser.add_argument("--skip_embedding", action="store_true", help="Skip DNABERT-2 embedding")

    args = parser.parse_args()
    run_preprocessing(
        training_file=args.training_file,
        output_dir=args.output_dir,
        max_pairs=args.max_pairs,
        device=args.device,
        dna_batch_size=args.dna_batch_size,
        skip_embedding=args.skip_embedding,
    )
