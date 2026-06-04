"""
Preprocessing pipeline for crisprSFM: CRISPR guide RNA off-target binding.

Downloads curated CRISPR off-target datasets, parses guide–off-target
pairs, embeds both with DNABERT-2, and saves CALM-ready tensors.

Data sources:
  - TrueOT (baolab-rice): 1,806 curated guide–off-target pairs from 35 gRNAs
    across 11 studies (includes GUIDE-seq and CHANGE-seq validated sites).
    Bao et al., direct download from GitHub.
  - Local files: any TSV/CSV with guide_seq, offtarget_seq columns

Usage:
    python -m calm.preprocess.crispr \
        --output_dir /path/to/output \
        --device cuda

    # Or with a local file:
    python -m calm.preprocess.crispr \
        --output_dir /path/to/output \
        --guideseq_file /path/to/my_data.tsv \
        --device cuda
"""

from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path

import pandas as pd
import requests

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 1. Download TrueOT curated dataset (baolab-rice, GitHub)
# ---------------------------------------------------------------------------

# TrueOT: 1,806 unique (gRNA, off-target, label) triplets from 35 gRNAs
# across 11 studies. Columns: gRNA, OT, label (1 = true off-target)
# Source: https://github.com/baolab-rice/CRISPR_OT_scoring
TRUEOT_URL = (
    "https://raw.githubusercontent.com/baolab-rice/CRISPR_OT_scoring/"
    "master/custom_scoring/parsed_datasets/TrueOT/"
    "TrueOT_1806uniqueTriplet_gRNA_OT_label.csv"
)


def fetch_trueot_data(
    max_pairs: int = 0,
) -> pd.DataFrame:
    """Download and parse the TrueOT curated off-target dataset.

    This dataset aggregates validated off-target sites from 11 studies
    including GUIDE-seq and CHANGE-seq experiments.

    Parameters
    ----------
    max_pairs : int
        Maximum pairs to keep (0 = all).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: guide_name, guide_seq, offtarget_seq, read_count, source
    """
    print("  Downloading TrueOT dataset from GitHub...")
    resp = requests.get(TRUEOT_URL, timeout=60)
    if resp.status_code != 200:
        print(
            f"  Warning: Could not download TrueOT data (HTTP {resp.status_code}).\n"
            f"  URL: {TRUEOT_URL}\n"
            f"  Please provide a local file with --guideseq_file instead."
        )
        return pd.DataFrame()

    df = pd.read_csv(io.StringIO(resp.text))
    print(f"  Downloaded {len(df)} rows, columns: {df.columns.tolist()}")

    # TrueOT format: gRNA, OT, label
    # Keep only positive off-targets (label == 1)
    if "label" in df.columns:
        n_before = len(df)
        df = df[df["label"] == 1].reset_index(drop=True)
        print(f"  Filtered to label==1 (true off-targets): {n_before} -> {len(df)} pairs")

    # Map to standard column names
    result = pd.DataFrame()

    # Detect guide column
    guide_col = None
    for c in ["gRNA", "grna", "guide_seq", "sgRNA_seq"]:
        if c in df.columns:
            guide_col = c
            break
    if guide_col is None:
        print(f"  Error: Could not find guide sequence column in: {df.columns.tolist()}")
        return pd.DataFrame()

    # Detect off-target column
    ot_col = None
    for c in ["OT", "ot", "offtarget_seq", "off_seq", "off_target_seq"]:
        if c in df.columns:
            ot_col = c
            break
    if ot_col is None:
        print(f"  Error: Could not find off-target sequence column in: {df.columns.tolist()}")
        return pd.DataFrame()

    result["guide_seq"] = df[guide_col].str.upper().str.strip()
    result["offtarget_seq"] = df[ot_col].str.upper().str.strip()
    result["read_count"] = 1  # TrueOT uses binary labels, not read counts
    result["guide_name"] = result["guide_seq"]  # Use sequence as name
    result["source"] = "trueot"

    # Remove rows with non-DNA characters
    dna_re = re.compile(r"^[ACGTNacgtn]+$")
    valid = (
        result["guide_seq"].apply(lambda s: bool(dna_re.match(s)) if pd.notna(s) else False)
        & result["offtarget_seq"].apply(lambda s: bool(dna_re.match(s)) if pd.notna(s) else False)
    )
    n_removed = len(result) - valid.sum()
    result = result[valid].reset_index(drop=True)
    if n_removed > 0:
        print(f"  Removed {n_removed} rows with invalid DNA characters")

    if max_pairs > 0:
        result = result.head(max_pairs)

    print(f"  Parsed {len(result)} validated off-target pairs from TrueOT")
    return result


# ---------------------------------------------------------------------------
# 2. Download CCLMoff curated dataset (Figshare, 82K off-targets)
# ---------------------------------------------------------------------------

# CCLMoff: 82,699 validated off-target sites from 418 sgRNAs across
# 13 detection technologies (GUIDE-seq, CHANGE-seq, CIRCLE-seq, etc.)
# Source: https://doi.org/10.6084/m9.figshare.27080566.v2
# Columns: sgRNA_seq, off_seq, label, id, sgRNA_type
CCLMOFF_URL = "https://ndownloader.figshare.com/files/49344577"
CCLMOFF_FILENAME = "09212024_CCLMoff_dataset.csv"


def fetch_cclmoff_data(
    local_file: str | None = None,
    max_pairs: int = 0,
) -> pd.DataFrame:
    """Download and parse the CCLMoff curated off-target dataset.

    This dataset aggregates 82,699 validated off-target sites from 418 sgRNAs
    across 13 genome-wide detection technologies including GUIDE-seq,
    CHANGE-seq, CIRCLE-seq, SITE-seq, and more.

    Parameters
    ----------
    local_file : str or None
        Path to local CCLMoff CSV file. If None, downloads from Figshare (~715 MB).
    max_pairs : int
        Maximum pairs to keep (0 = all).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: guide_name, guide_seq, offtarget_seq, read_count, source
    """
    if local_file:
        csv_path = Path(local_file)
    else:
        # Check if already downloaded to output dir
        csv_path = None

    if csv_path and csv_path.exists():
        print(f"  Loading CCLMoff data from {csv_path}...")
    else:
        print(f"  Downloading CCLMoff dataset from Figshare (~715 MB)...")
        print(f"  URL: {CCLMOFF_URL}")
        try:
            resp = requests.get(CCLMOFF_URL, timeout=600, stream=True)
            if resp.status_code != 200:
                print(
                    f"  Warning: Could not download CCLMoff data (HTTP {resp.status_code}).\n"
                    f"  Download manually from: https://doi.org/10.6084/m9.figshare.27080566.v2\n"
                    f"  Then pass with --cclmoff_file"
                )
                return pd.DataFrame()
            # Save to a temp path
            csv_path = Path(local_file) if local_file else Path(CCLMOFF_FILENAME)
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(csv_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192 * 128):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0 and downloaded % (50 * 1024 * 1024) < len(chunk):
                        pct = downloaded * 100 // total
                        print(f"    Downloaded {downloaded // (1024*1024)} MB / {total // (1024*1024)} MB ({pct}%)")
            print(f"  Downloaded {downloaded // (1024*1024)} MB to {csv_path}")
        except Exception as e:
            print(f"  Error downloading CCLMoff: {e}")
            print(f"  Download manually from: https://doi.org/10.6084/m9.figshare.27080566.v2")
            return pd.DataFrame()

    # Parse the CSV
    print(f"  Parsing CCLMoff CSV...")
    df = pd.read_csv(csv_path, dtype=str)
    print(f"  Loaded {len(df)} rows, columns: {df.columns.tolist()}")

    # CCLMoff format: sgRNA_seq, off_seq, label, id, sgRNA_type
    # Keep only positive off-targets (label == 1)
    if "label" in df.columns:
        # Show unique label values for debugging
        unique_labels = df["label"].unique()[:10]
        print(f"  Label values found: {unique_labels.tolist()}")
        n_before = len(df)
        # Handle both "1" and "1.0" string representations
        label_numeric = pd.to_numeric(df["label"], errors="coerce")
        df = df[label_numeric == 1].reset_index(drop=True)
        print(f"  Filtered to label==1 (true off-targets): {n_before} -> {len(df)} pairs")

    result = pd.DataFrame()

    # Map columns
    if "sgRNA_seq" in df.columns:
        result["guide_seq"] = df["sgRNA_seq"].str.upper().str.strip()
    else:
        print(f"  Error: 'sgRNA_seq' column not found in CCLMoff data")
        return pd.DataFrame()

    if "off_seq" in df.columns:
        result["offtarget_seq"] = df["off_seq"].str.upper().str.strip()
    else:
        print(f"  Error: 'off_seq' column not found in CCLMoff data")
        return pd.DataFrame()

    # Use read count if available, otherwise binary
    if "read" in df.columns:
        result["read_count"] = pd.to_numeric(df["read"], errors="coerce").fillna(1).astype(int)
    else:
        result["read_count"] = 1

    # Use sgRNA_type as guide name (groups guides by identity)
    if "sgRNA_type" in df.columns:
        result["guide_name"] = df["sgRNA_type"].str.strip()
    else:
        result["guide_name"] = result["guide_seq"]

    result["source"] = "cclmoff"

    # Remove rows with non-DNA characters or underscores (bulge notation)
    dna_re = re.compile(r"^[ACGTNacgtn]+$")
    valid_guide = result["guide_seq"].apply(lambda s: bool(dna_re.match(str(s))) if pd.notna(s) else False).astype(bool)
    valid_ot = result["offtarget_seq"].apply(lambda s: bool(dna_re.match(str(s))) if pd.notna(s) else False).astype(bool)
    valid = valid_guide & valid_ot
    n_removed = len(result) - valid.sum()
    result = result[valid].reset_index(drop=True)
    if n_removed > 0:
        print(f"  Removed {n_removed} rows with non-standard DNA characters (bulges, etc.)")

    if max_pairs > 0:
        result = result.head(max_pairs)

    print(f"  Parsed {len(result)} validated off-target pairs from CCLMoff")
    return result


# ---------------------------------------------------------------------------
# 3. Load local off-target data files
# ---------------------------------------------------------------------------

# Valid DNA characters
_DNA_RE = re.compile(r"^[ACGTNacgtn]+$")


def fetch_local_data(
    local_file: str,
    source_label: str = "local",
    max_pairs: int = 0,
) -> pd.DataFrame:
    """Load and parse a local CRISPR off-target data file.

    Supports TSV/CSV with auto-detected column names.

    Parameters
    ----------
    local_file : str
        Path to local TSV/CSV file.
    source_label : str
        Label for the data source.
    max_pairs : int
        Maximum pairs to keep (0 = all).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: guide_name, guide_seq, offtarget_seq, read_count, source
    """
    print(f"  Loading data from {local_file}...")
    raw = Path(local_file).read_text()
    return _parse_offtarget_tsv(raw, source=source_label, max_pairs=max_pairs)


# ---------------------------------------------------------------------------
# 3. Parse off-target TSV files
# ---------------------------------------------------------------------------

# Valid DNA characters (allows IUPAC ambiguity codes)
_DNA_RE = re.compile(r"^[ACGTNacgtn]+$")


def _parse_offtarget_tsv(
    raw_text: str,
    source: str,
    max_pairs: int = 0,
) -> pd.DataFrame:
    """Parse a TSV file of CRISPR off-target sites into a uniform format.

    Supports multiple TSV formats by auto-detecting column names.
    Expected columns (case-insensitive, flexible naming):
      - Guide/target sequence (~20nt): "guide_seq", "targetSequence", "spacer", etc.
      - Off-target site sequence: "offtarget_seq", "Site_Sequence", "Genomic", etc.
      - Read count (optional): "read_count", "GUIDE-SEQ Reads", "CHANGE-seq_reads", etc.
      - Guide name (optional): "guide_name", "targetName", "sgRNA_Name", etc.

    Parameters
    ----------
    raw_text : str
        Raw TSV/CSV content.
    source : str
        Data source label ("guideseq" or "changeseq").
    max_pairs : int
        Maximum pairs (0 = all).

    Returns
    -------
    pd.DataFrame
    """
    # Auto-detect delimiter
    if "\t" in raw_text[:500]:
        sep = "\t"
    else:
        sep = ","

    df = pd.read_csv(io.StringIO(raw_text), sep=sep, dtype=str)
    df.columns = df.columns.str.strip()

    # Map column names to standard names
    col_map = _detect_columns(df.columns.tolist())
    if col_map is None:
        print(f"  Warning: Could not auto-detect columns in {source} data.")
        print(f"  Found columns: {df.columns.tolist()}")
        print(f"  Expected: guide_seq/targetSequence, offtarget_seq/Site_Sequence, read_count")
        return pd.DataFrame()

    result = pd.DataFrame()
    result["guide_seq"] = df[col_map["guide_seq"]].str.upper().str.strip()
    result["offtarget_seq"] = df[col_map["offtarget_seq"]].str.upper().str.strip()

    if col_map.get("read_count"):
        result["read_count"] = pd.to_numeric(df[col_map["read_count"]], errors="coerce").fillna(0).astype(int)
    else:
        result["read_count"] = 1

    if col_map.get("guide_name"):
        result["guide_name"] = df[col_map["guide_name"]].str.strip()
    else:
        result["guide_name"] = result["guide_seq"]

    result["source"] = source

    # Clean: remove rows with non-DNA characters or empty sequences
    valid_guide = result["guide_seq"].apply(lambda s: bool(_DNA_RE.match(s)) if pd.notna(s) else False)
    valid_ot = result["offtarget_seq"].apply(lambda s: bool(_DNA_RE.match(s)) if pd.notna(s) else False)
    n_before = len(result)
    result = result[valid_guide & valid_ot].reset_index(drop=True)
    n_removed = n_before - len(result)
    if n_removed > 0:
        print(f"  Removed {n_removed} rows with invalid DNA characters")

    if max_pairs > 0:
        result = result.head(max_pairs)

    print(f"  Parsed {len(result)} off-target pairs from {source}")
    return result


def _detect_columns(columns: list[str]) -> dict[str, str] | None:
    """Auto-detect column mapping from various CRISPR off-target TSV formats."""
    col_lower = {c.lower().replace(" ", "_").replace("-", "_"): c for c in columns}

    guide_candidates = [
        "targetsequence", "guide_seq", "guide_sequence", "spacer",
        "sgrna_seq", "sgrna_sequence", "target_seq", "crRNA",
    ]
    offtarget_candidates = [
        "site_sequence", "offtarget_seq", "off_target_seq", "offtarget_sequence",
        "genomic_sequence", "off_target_sequence", "site_seq",
    ]
    readcount_candidates = [
        "read_count", "guide_seq_reads", "changeseq_reads", "change_seq_reads",
        "reads", "count", "bi.sum.mi", "bi.sum.all",
    ]
    name_candidates = [
        "targetname", "guide_name", "sgrna_name", "target_name", "name",
    ]

    mapping = {}

    for cand in guide_candidates:
        if cand in col_lower:
            mapping["guide_seq"] = col_lower[cand]
            break
    for cand in offtarget_candidates:
        if cand in col_lower:
            mapping["offtarget_seq"] = col_lower[cand]
            break
    for cand in readcount_candidates:
        if cand in col_lower:
            mapping["read_count"] = col_lower[cand]
            break
    for cand in name_candidates:
        if cand in col_lower:
            mapping["guide_name"] = col_lower[cand]
            break

    if "guide_seq" not in mapping or "offtarget_seq" not in mapping:
        return None

    return mapping


# ---------------------------------------------------------------------------
# 4. Process and merge datasets
# ---------------------------------------------------------------------------


def process_offtarget_data(
    combined: pd.DataFrame,
    min_read_count: int = 1,
) -> pd.DataFrame:
    """Filter and deduplicate off-target data.

    Parameters
    ----------
    combined : pd.DataFrame
        Combined off-target pairs with columns: guide_seq, offtarget_seq,
        read_count, guide_name, source.
    min_read_count : int
        Minimum read count to keep a pair (filters low-confidence off-targets).

    Returns
    -------
    pd.DataFrame
        Filtered, deduplicated DataFrame.
    """
    print(f"  Combined: {len(combined)} total pairs")

    # Filter by read count
    if min_read_count > 1:
        n_before = len(combined)
        combined = combined[combined["read_count"] >= min_read_count].reset_index(drop=True)
        print(f"  Filtered by read_count >= {min_read_count}: {n_before} -> {len(combined)} pairs")

    # Deduplicate by (guide_seq, offtarget_seq)
    n_before = len(combined)
    combined = combined.drop_duplicates(subset=["guide_seq", "offtarget_seq"], keep="first")
    combined = combined.reset_index(drop=True)
    if n_before != len(combined):
        print(f"  Deduplicated: {n_before} -> {len(combined)} unique pairs")

    # Convert RNA to DNA (U -> T) for DNABERT-2 compatibility
    combined["guide_seq"] = combined["guide_seq"].str.replace("U", "T")
    combined["offtarget_seq"] = combined["offtarget_seq"].str.replace("U", "T")

    n_guides = combined["guide_name"].nunique()
    print(f"  Final: {len(combined)} pairs across {n_guides} unique guides")

    return combined


# ---------------------------------------------------------------------------
# 5. Build metadata and splits
# ---------------------------------------------------------------------------


def build_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Build metadata DataFrame for CALM training pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Processed off-target data with guide_seq, offtarget_seq, guide_name columns.

    Returns
    -------
    pd.DataFrame
        Metadata with ag_id, ab_id, cluster_id, and hash columns.
    """
    # Map unique guide sequences to IDs (agent = guide RNA)
    guide_seqs = df["guide_seq"].unique().tolist()
    guide_to_id = {seq: i for i, seq in enumerate(guide_seqs)}

    # Map unique off-target sequences to IDs (target = off-target DNA)
    ot_seqs = df["offtarget_seq"].unique().tolist()
    ot_to_id = {seq: i for i, seq in enumerate(ot_seqs)}

    # Cluster by guide identity for OOD splitting
    # All off-targets for the same guide go in the same split
    guide_names = df["guide_name"].unique().tolist()
    guidename_to_cluster = {name: i for i, name in enumerate(guide_names)}

    # Hard negatives (label=0) get unique negative IDs so they are NEVER
    # grouped as positives with any other entry. They appear in batches
    # only as in-batch negatives, teaching the model that sequence
    # similarity alone doesn't mean binding.
    has_labels = "label" in df.columns
    neg_ag_counter = -1  # Negative IDs for hard negatives
    neg_ab_counter = -1

    records = []
    for i, row in df.iterrows():
        is_hard_neg = has_labels and int(row.get("label", 1)) == 0

        if is_hard_neg:
            # Unique IDs that won't match any positive entry
            ag_id = neg_ag_counter
            ab_id = neg_ab_counter
            neg_ag_counter -= 1
            neg_ab_counter -= 1
        else:
            ag_id = guide_to_id[row["guide_seq"]]
            ab_id = ot_to_id[row["offtarget_seq"]]

        records.append({
            "pair_index": i,
            "guide_name": row["guide_name"],
            "guide_seq": row["guide_seq"],
            "offtarget_seq": row["offtarget_seq"],
            "read_count": row["read_count"],
            "source": row["source"],
            "ag_id": ag_id,
            "ab_id": ab_id,
            "cluster_id": guidename_to_cluster[row["guide_name"]],
            "Unique_ag_vh_vl_hash": f"guide_{ag_id}_{i}",
            "label": 0 if is_hard_neg else 1,
        })

    meta = pd.DataFrame(records)
    n_pos = (meta["label"] == 1).sum()
    n_neg = (meta["label"] == 0).sum()
    print(f"  Metadata: {len(meta)} pairs ({n_pos} positive, {n_neg} hard negatives)")
    print(f"  Clusters (guides): {meta['cluster_id'].nunique()}")
    print(f"  Unique guide sequences: {meta[meta['label']==1]['ag_id'].nunique()}")
    print(f"  Unique off-target sequences: {meta[meta['label']==1]['ab_id'].nunique()}")
    return meta


def build_splits(
    df: pd.DataFrame,
    output_dir: Path,
    n_folds: int = 5,
    seed: int = 42,
) -> None:
    """Create train/val/test split index files based on guide RNA clusters.

    Same cluster-based OOD splitting strategy as CALM / tSFM:
    all off-targets for the same guide go in the same split.

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
# 6. Embed sequences with DNABERT-2
# ---------------------------------------------------------------------------


def embed_dna_dnabert2(
    sequences: list[str],
    model_name: str = "zhihan1996/DNABERT-2-117M",
    device: str = "cpu",
    batch_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Embed DNA sequences using DNABERT-2.

    Replicates the same embedding logic as jaspar.py to ensure
    consistency across SFM domains.

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
# 7. Main pipeline
# ---------------------------------------------------------------------------


def run_preprocessing(
    output_dir: str,
    guideseq_file: str | None = None,
    changeseq_file: str | None = None,
    cclmoff_file: str | None = None,
    use_cclmoff: bool = False,
    exclude_guides_file: str | None = None,
    hard_negatives_file: str | None = None,
    min_read_count: int = 1,
    max_pairs: int = 0,
    device: str = "cpu",
    dna_batch_size: int = 16,
    skip_embedding: bool = False,
) -> None:
    """Run the full CRISPR off-target preprocessing pipeline.

    Parameters
    ----------
    output_dir : str
        Directory to save all output files.
    guideseq_file : str or None
        Path to local TSV/CSV with additional GUIDE-seq off-target data.
    changeseq_file : str or None
        Path to local TSV/CSV with additional CHANGE-seq off-target data.
    cclmoff_file : str or None
        Path to local CCLMoff CSV file. If None and --use_cclmoff is set,
        downloads from Figshare (~715 MB).
    use_cclmoff : bool
        Whether to download and use the CCLMoff dataset.
    exclude_guides_file : str or None
        Path to text file with guide sequences to exclude (one per line).
        Used to hold out validation guides (e.g., CRISPRoffT overlap).
    hard_negatives_file : str or None
        Path to CSV with hard negative pairs (guide_seq, offtarget_seq, label=0).
        These get unique IDs so they serve purely as in-batch negatives.
    min_read_count : int
        Minimum read count to keep a pair.
    max_pairs : int
        Maximum total pairs (0 = all). Useful for smoke tests.
    device : str
        Device for DNABERT-2 inference.
    dna_batch_size : int
        Batch size for DNABERT-2 inference.
    skip_embedding : bool
        If True, skip embedding step (use if tensors already exist).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_dfs = []

    # --- Step 1a: CCLMoff dataset (large, 82K off-targets) ---
    if use_cclmoff or cclmoff_file:
        cclmoff_cache = out / "cclmoff_raw.csv"
        if cclmoff_cache.exists():
            print("Step 1a: Loading cached CCLMoff data...")
            cclmoff_df = pd.read_csv(cclmoff_cache)
        else:
            print("Step 1a: Fetching CCLMoff dataset...")
            cclmoff_df = fetch_cclmoff_data(
                local_file=cclmoff_file or str(out / CCLMOFF_FILENAME),
                max_pairs=0,  # Don't limit during download; limit later
            )
            if len(cclmoff_df) > 0:
                cclmoff_df.to_csv(cclmoff_cache, index=False)
        if len(cclmoff_df) > 0:
            all_dfs.append(cclmoff_df)
    else:
        # --- Step 1a (fallback): TrueOT curated dataset (small, 250 pairs) ---
        trueot_cache = out / "trueot_raw.csv"
        if trueot_cache.exists():
            print("Step 1a: Loading cached TrueOT data...")
            trueot_df = pd.read_csv(trueot_cache)
        else:
            print("Step 1a: Fetching TrueOT curated dataset...")
            trueot_df = fetch_trueot_data(max_pairs=max_pairs)
            if len(trueot_df) > 0:
                trueot_df.to_csv(trueot_cache, index=False)
        if len(trueot_df) > 0:
            all_dfs.append(trueot_df)

    # --- Step 1b: Load local GUIDE-seq file if provided ---
    if guideseq_file:
        guideseq_cache = out / "guideseq_raw.csv"
        if guideseq_cache.exists():
            print("Step 1b: Loading cached GUIDE-seq data...")
            guideseq_df = pd.read_csv(guideseq_cache)
        else:
            print("Step 1b: Loading local GUIDE-seq data...")
            guideseq_df = fetch_local_data(guideseq_file, source_label="guideseq", max_pairs=max_pairs)
            if len(guideseq_df) > 0:
                guideseq_df.to_csv(guideseq_cache, index=False)
        if len(guideseq_df) > 0:
            all_dfs.append(guideseq_df)

    # --- Step 1c: Load local CHANGE-seq file if provided ---
    if changeseq_file:
        changeseq_cache = out / "changeseq_raw.csv"
        if changeseq_cache.exists():
            print("Step 1c: Loading cached CHANGE-seq data...")
            changeseq_df = pd.read_csv(changeseq_cache)
        else:
            print("Step 1c: Loading local CHANGE-seq data...")
            changeseq_df = fetch_local_data(changeseq_file, source_label="changeseq", max_pairs=max_pairs)
            if len(changeseq_df) > 0:
                changeseq_df.to_csv(changeseq_cache, index=False)
        if len(changeseq_df) > 0:
            all_dfs.append(changeseq_df)

    # --- Step 2: Merge and process ---
    if not all_dfs:
        raise ValueError(
            "No off-target data found. Use --use_cclmoff to download the CCLMoff dataset,\n"
            "or provide a local file with --guideseq_file, --changeseq_file, or --cclmoff_file."
        )

    print("Step 2: Merging and processing off-target data...")
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = process_offtarget_data(combined, min_read_count=min_read_count)

    # --- Step 2b: Exclude validation guides (e.g., CRISPRoffT overlap) ---
    if exclude_guides_file:
        excl_path = Path(exclude_guides_file)
        if excl_path.exists():
            excl_seqs = set(
                line.strip().upper()
                for line in excl_path.read_text().splitlines()
                if line.strip()
            )
            n_before = len(combined)
            combined = combined[
                ~combined["guide_seq"].str.upper().str.strip().isin(excl_seqs)
            ].reset_index(drop=True)
            n_removed = n_before - len(combined)
            print(f"  Excluded {n_removed} pairs from {len(excl_seqs)} validation guide sequences")
            print(f"  Remaining: {len(combined)} pairs across {combined['guide_name'].nunique()} guides")
        else:
            print(f"  Warning: exclude_guides_file not found: {excl_path}")

    if max_pairs > 0:
        combined = combined.head(max_pairs)
        print(f"  Capped to {max_pairs} pairs (--max_pairs)")

    # --- Step 2c: Add hard negatives if provided ---
    if hard_negatives_file:
        hn_path = Path(hard_negatives_file)
        if hn_path.exists():
            print("Step 2c: Loading hard negatives...")
            hn_df = pd.read_csv(hn_path, dtype=str)
            hn_df = hn_df[["guide_seq", "offtarget_seq", "guide_name"]].copy()
            hn_df["read_count"] = "0"
            hn_df["source"] = "hard_negative"
            hn_df["label"] = 0
            # Mark positive data
            combined["label"] = 1
            combined = pd.concat([combined, hn_df], ignore_index=True)
            print(f"  Added {len(hn_df)} hard negatives ({hn_df['guide_seq'].nunique()} guides)")
            print(f"  Total: {len(combined)} pairs ({(combined['label']==1).sum()} positive, "
                  f"{(combined['label']==0).sum()} negative)")
        else:
            print(f"  Warning: hard_negatives_file not found: {hn_path}")
            combined["label"] = 1
    else:
        combined["label"] = 1

    # --- Step 4: Build metadata ---
    print("Step 3: Building metadata...")
    df = build_metadata(combined)
    df.to_csv(out / "metadata.csv", index=False)

    # --- Step 5: Embed sequences ---
    if not skip_embedding:
        guide_seqs = df["guide_seq"].tolist()
        offtarget_seqs = df["offtarget_seq"].tolist()

        print("Step 4a: Embedding guide RNA sequences with DNABERT-2...")
        ag_embed, ag_mask = embed_dna_dnabert2(
            guide_seqs, device=device, batch_size=dna_batch_size,
        )
        torch.save(ag_embed, out / "ag_embed.pt")
        torch.save(ag_mask, out / "ag_mask.pt")
        print(f"  Saved ag_embed.pt {ag_embed.shape} and ag_mask.pt {ag_mask.shape}")

        print("Step 4b: Embedding off-target DNA sequences with DNABERT-2...")
        ab_embed, ab_mask = embed_dna_dnabert2(
            offtarget_seqs, device=device, batch_size=dna_batch_size,
        )
        torch.save(ab_embed, out / "ab_embed.pt")
        torch.save(ab_mask, out / "ab_mask.pt")
        print(f"  Saved ab_embed.pt {ab_embed.shape} and ab_mask.pt {ab_mask.shape}")
    else:
        print("Step 4: Skipping embedding (--skip_embedding set)")

    # --- Step 6: Build train/val/test splits ---
    print("Step 5: Building train/val/test splits...")
    # CALM expects: data_dir/split_index/{split_method}/
    split_dir = out / "split_index" / "by_cluster_test_cv5f"
    build_splits(df, split_dir)

    print(f"\nDone! All outputs saved to {out}")
    print(f"  metadata.csv:  {len(df)} guide-offtarget pairs")
    print(f"  ag_embed.pt:   guide RNA embeddings (DNABERT-2, 768-dim)")
    print(f"  ab_embed.pt:   off-target DNA embeddings (DNABERT-2, 768-dim)")
    print(f"  ag_mask.pt:    guide RNA padding masks")
    print(f"  ab_mask.pt:    off-target DNA padding masks")
    print(f"  splits/:       CV split index files")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess CRISPR off-target data for crisprSFM training"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save preprocessed outputs",
    )
    parser.add_argument(
        "--guideseq_file",
        type=str,
        default=None,
        help="Path to local GUIDE-seq TSV file (if not set, downloads from GitHub)",
    )
    parser.add_argument(
        "--changeseq_file",
        type=str,
        default=None,
        help="Path to local CHANGE-seq TSV file",
    )
    parser.add_argument(
        "--cclmoff_file",
        type=str,
        default=None,
        help="Path to local CCLMoff CSV file (if not set, use --use_cclmoff to download)",
    )
    parser.add_argument(
        "--use_cclmoff",
        action="store_true",
        help="Download and use the CCLMoff dataset (~715 MB, 82K off-targets)",
    )
    parser.add_argument(
        "--exclude_guides_file",
        type=str,
        default=None,
        help="Text file with guide sequences to exclude (one per line), for validation holdout",
    )
    parser.add_argument(
        "--hard_negatives_file",
        type=str,
        default=None,
        help="CSV with hard negative pairs (guide_seq, offtarget_seq, guide_name, label=0)",
    )
    parser.add_argument(
        "--min_read_count",
        type=int,
        default=1,
        help="Minimum read count to keep a pair (default: 1 = keep all)",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=0,
        help="Max pairs to process (0 = all, use small number for testing)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for DNABERT-2 inference (cpu or cuda)",
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
    args = parser.parse_args()

    run_preprocessing(
        output_dir=args.output_dir,
        guideseq_file=args.guideseq_file,
        changeseq_file=args.changeseq_file,
        cclmoff_file=args.cclmoff_file,
        use_cclmoff=args.use_cclmoff,
        exclude_guides_file=args.exclude_guides_file,
        hard_negatives_file=args.hard_negatives_file,
        min_read_count=args.min_read_count,
        max_pairs=args.max_pairs,
        device=args.device,
        dna_batch_size=args.dna_batch_size,
        skip_embedding=args.skip_embedding,
    )
