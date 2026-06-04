"""Utility functions for CALM dataset preprocessing, including ID generation, mask processing, sampling, and split construction."""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from calm.preprocess.cli import PreprocessConfig


def create_ids_from_seqs(seqs: list[str]) -> list[int]:
    """Create stable integer IDs from sequence strings preserving first-seen order.

    Parameters
    ----------
    seqs : list[str]
        Input sequence strings.

    Returns
    -------
    list[int]
        Stable integer IDs aligned with ``seqs``.
    """
    unique_seqs = list(dict.fromkeys(seqs))
    seq_to_id = {seq: idx for idx, seq in enumerate(unique_seqs)}
    return [seq_to_id[seq] for seq in seqs]


def _extract_mask_values(mask_str: str) -> list[int]:
    """Extract binary mask values from a string.

    Parameters
    ----------
    mask_str : str
        String containing mask-like characters.

    Returns
    -------
    list[int]
        List of parsed binary values (0/1).
    """
    return list(map(int, mask_str))


def add_seq_id_fields(df: pd.DataFrame, config: PreprocessConfig) -> pd.DataFrame:
    """Add unique antibody and antigen sequence IDs to a DataFrame containing sequence data.

    Parameters
    ----------
    df : pd.DataFrame
        Metadata DataFrame
    config: PreprocessConfig
        Preprocessing configuration

    Returns
    -------
    pd.DataFrame
        The DataFrame with added 'ab_id' and 'ag_id' columns.
    """
    col_ag_aa = config.col_ag_aa
    col_vh_aa = config.col_vh_aa
    col_vl_aa = config.col_vl_aa
    col_ag_id = config.col_ag_id
    col_ab_id = config.col_ab_id

    # Store original shape and index to ensure no rows are lost
    original_shape = df.shape
    original_index = df.index.copy()

    # Create a copy to work with to ensure original data is not modified
    df_work = df.copy()

    # Generate ab_id based on uniqueness of (VH, VL) tuples
    ab_tuples = list(zip(df_work[col_vh_aa], df_work[col_vl_aa], strict=False))
    # Use dict.fromkeys to preserve first occurrence order and remove duplicates
    unique_ab_tuples = list(dict.fromkeys(ab_tuples))
    ab_id_mapping = {tup: idx for idx, tup in enumerate(unique_ab_tuples)}
    df_work[col_ab_id] = [ab_id_mapping[tup] for tup in ab_tuples]

    # Generate ag_id based on uniqueness of antigen sequences
    # Use drop_duplicates to preserve first occurrence order
    unique_ag_df = (
        df_work[[col_ag_aa]].drop_duplicates(keep="first").reset_index(drop=True)
    )
    ag_id_mapping = {seq: idx for idx, seq in enumerate(unique_ag_df[col_ag_aa])}
    df_work[col_ag_id] = df_work[col_ag_aa].map(ag_id_mapping)

    # Verify that original columns are unchanged
    original_cols = df.columns.tolist()
    for col in original_cols:
        if not df_work[col].equals(df[col]):
            raise RuntimeError(f"Column {col} was modified during processing")

    # Verify that row order is preserved
    if not df_work.index.equals(original_index):
        raise RuntimeError("Row order was changed during processing")

    # Verify no rows were added or removed
    if df_work.shape[0] != original_shape[0]:
        raise RuntimeError(
            f"Number of rows changed: {original_shape[0]} -> {df_work.shape[0]}"
        )

    # Verify no NaN values were introduced in the ID columns
    if df_work[config.col_ab_id].isna().any():
        raise ValueError("NaN values found in ab_id column")
    if df_work[config.col_ag_id].isna().any():
        raise ValueError("NaN values found in ag_id column")

    # Only add the new ID columns to the original dataframe
    df[col_ab_id] = df_work[col_ab_id]
    df[col_ag_id] = df_work[col_ag_id]

    # Final verification that we still have the same number of rows
    if df.shape[0] != original_shape[0]:
        raise RuntimeError("Final dataframe lost rows during ID addition")

    # Validation: Check for ab_id collisions (should be one-to-one with (VH, VL) tuple)
    ab_tuples = list(zip(df[col_vh_aa], df[col_vl_aa], strict=False))
    ab_id_to_tuple: dict[str, tuple[str, str] | tuple[None, None]] = {}
    collision_found = False
    for idx, tup in enumerate(ab_tuples):
        ab_id = df[col_ab_id].iloc[idx]
        if ab_id in ab_id_to_tuple:
            if ab_id_to_tuple[ab_id] != tup:
                print(
                    f"Collision: ab_id {ab_id} maps to multiple tuples: {ab_id_to_tuple[ab_id]} and {tup}"
                )
                collision_found = True
        else:
            ab_id_to_tuple[ab_id] = tup
    if not collision_found:
        print(
            "No ab_id collisions detected. Each ab_id maps to a unique (VH, VL) tuple."
        )

    # Validation: Check for ag_id collisions (should be one-to-one with antigen sequence)
    ag_id_to_seq: dict[str, str] = {}
    ag_collision_found = False
    for idx, seq in enumerate(df[col_ag_aa]):
        ag_id = df[col_ag_id].iloc[idx]
        if ag_id in ag_id_to_seq:
            if ag_id_to_seq[ag_id] != seq:
                print(
                    f"Collision: ag_id {ag_id} maps to multiple sequences: {ag_id_to_seq[ag_id]} and {seq}"
                )
                ag_collision_found = True
        else:
            ag_id_to_seq[ag_id] = seq
    if not ag_collision_found:
        print(
            "No ag_id collisions detected. Each ag_id maps to a unique antigen sequence."
        )

    return df


def add_epitope_paratope_fields(
    df: pd.DataFrame, config: PreprocessConfig
) -> pd.DataFrame:
    """Add epitope and paratope sequence and ID fields to the metadata DataFrame based on the provided masks.

    Parameters
    ----------
    df : pd.DataFrame
        Metadata DataFrame
    config : PreprocessConfig
        Preprocessing configuration

    Returns
    -------
    pd.DataFrame
        Metadata DataFrame with added epitope and paratope fields
    """
    col_ag_aa = config.col_ag_aa
    col_vh_aa = config.col_vh_aa
    col_vl_aa = config.col_vl_aa
    col_ep_mask = config.col_ep_mask
    col_pa_vh_mask = config.col_pa_vh_mask
    col_pa_vl_mask = config.col_pa_vl_mask
    col_ep_aa = config.col_ep_aa
    col_pa_aa = config.col_pa_aa
    col_ep_id = config.col_ep_id
    col_pa_id = config.col_pa_id
    sep = config.sep

    # Generate epitope sequences
    epitope_seqs = []
    for _, row in df.iterrows():
        ag_seq = str(row[col_ag_aa]) if not pd.isna(row[col_ag_aa]) else ""

        epitope_mask = str(row[col_ep_mask]) if not pd.isna(row[col_ep_mask]) else ""
        epitope_seq = ""

        sep_inserted = False
        for i, mask_char in enumerate(epitope_mask):
            if i >= len(ag_seq):
                break
            if mask_char == "1":
                epitope_seq += ag_seq[i]
                sep_inserted = False
            else:
                if not sep_inserted:
                    epitope_seq += sep
                    sep_inserted = True

        epitope_seqs.append(epitope_seq)

    # Generate paratope sequences
    paratope_seqs = []
    for _, row in df.iterrows():
        ab_vh_seq = str(row[col_vh_aa]) if not pd.isna(row[col_vh_aa]) else ""
        ab_vl_seq = str(row[col_vl_aa]) if not pd.isna(row[col_vl_aa]) else ""
        ab_seq = ab_vh_seq + ab_vl_seq

        paratope_vh_mask = (
            str(row[col_pa_vh_mask]) if not pd.isna(row[col_pa_vh_mask]) else ""
        )
        paratope_vl_mask = (
            str(row[col_pa_vl_mask]) if not pd.isna(row[col_pa_vl_mask]) else ""
        )
        paratope_mask = paratope_vh_mask + paratope_vl_mask
        paratope_seq = ""
        sep_inserted = False
        for i, mask_char in enumerate(paratope_mask):
            if i >= len(ab_seq):
                break
            if mask_char == "1":
                paratope_seq += ab_seq[i]
                sep_inserted = False
            else:
                if not sep_inserted:
                    paratope_seq += sep
                    sep_inserted = True

        paratope_seqs.append(paratope_seq)

    df[col_ep_aa] = epitope_seqs
    df[col_pa_aa] = paratope_seqs
    df[col_ep_id] = create_ids_from_seqs(epitope_seqs)
    df[col_pa_id] = create_ids_from_seqs(paratope_seqs)

    return df


def update_metadata(df: pd.DataFrame, config: PreprocessConfig) -> pd.DataFrame:
    """Update metadata columns after validating identity with reference metadata.

    Parameters
    ----------
    df : pd.DataFrame
        Generated metadata DataFrame.
    config : PreprocessConfig
        Preprocessing configuration containing reference file and column names.

    Returns
    -------
    pd.DataFrame
        Updated metadata DataFrame.
    """
    metadata_split_src_file = config.metadata_split_src_file
    col_update = config.col_cluster_input

    df_src = pd.read_csv(metadata_split_src_file)
    if not set(col_update).issubset(df.columns):
        raise ValueError(
            f"Cluster columns {col_update} not all found in metadata file columns: {df.columns.tolist()}"
        )
    if not set(col_update).issubset(df_src.columns):
        raise ValueError(
            f"Cluster columns {col_update} not all found in reference metadata file columns: {df_src.columns.tolist()}"
        )

    if len(df) != len(df_src):
        raise ValueError(
            f"Metadata length {len(df)} does not match reference {len(df_src)}"
        )

    # Check consistency between two metadata files
    identity_cols = [config.col_hash]
    is_identical = True
    for col in identity_cols:
        if col not in df.columns:
            raise KeyError(f"Column {col} not found in base metadata.")
        if col not in df_src.columns:
            raise KeyError(f"Column {col} not found in reference metadata.")

        if not df[col].equals(df_src[col]):
            is_identical = False
            raise ValueError(
                f"Column {col} does not match between generated and reference metadata."
            )

    if is_identical:
        for col in col_update:
            df[col] = df_src[col]

    return df


def add_cluster_id_fields(df: pd.DataFrame, config: PreprocessConfig) -> pd.DataFrame:
    """Add a cluster identifier column from one or more source cluster columns.

    Parameters
    ----------
    df : pd.DataFrame
        Metadata DataFrame.
    config : PreprocessConfig
        Preprocessing configuration containing input/output cluster column names.

    Returns
    -------
    pd.DataFrame
        Metadata DataFrame with the cluster ID output column added.
    """
    col_cluster_input = config.col_cluster_input
    col_cluster_output = config.col_cluster_output

    if not set(col_cluster_input).issubset(df.columns):
        raise ValueError(
            f"Cluster columns {col_cluster_input} not all found in metadata file columns: {df.columns.tolist()}"
        )

    if len(col_cluster_input) == 0:
        raise ValueError("col_cluster_input must contain at least one column name")
    if len(col_cluster_input) == 1:
        df[col_cluster_output] = df[col_cluster_input[0]].astype(str)
    else:
        df[col_cluster_output] = df[col_cluster_input].astype(str).agg("_".join, axis=1)

    print(
        f"Added {col_cluster_output} to metadata dataframe by combining {len(col_cluster_input)} cluster column(s)."
    )
    return df


def generate_masks(file_embed: Path, file_mask: Path) -> None:
    """Generate a padding mask tensor from an embedding tensor file.

    Parameters
    ----------
    file_embed : Path
        Path to input embedding tensor file.
    file_mask : Path
        Path to output mask tensor file.

    Returns
    -------
    None
        Writes mask tensor to ``file_mask``.
    """
    embed = torch.load(file_embed, weights_only=True)
    embed_mask = (embed != 0).any(dim=-1)
    torch.save(embed_mask, file_mask)


def generate_paratope_cdr_masks(config: PreprocessConfig) -> None:
    """Generate paratope mask and optional CDR masks for antibody sequences.

    Parameters
    ----------
    config : PreprocessConfig
        Preprocessing configuration containing metadata paths, output paths,
        and CDR mask specification fields.

    Returns
    -------
    None
        Writes paratope mask and, when available, CDR mask files to disk.
    """
    metadata_file = config.metadata_file
    dataset_size = config.dataset_size
    max_length = config.length_ab

    col_pa_vh_mask = config.col_pa_vh_mask
    col_pa_vl_mask = config.col_pa_vl_mask
    col_vh_aa = config.col_vh_aa
    paratope_mask_file = config.paratope_mask_file
    output_dir = config.input_dir

    # Initialize CDR masks if available
    is_cdr_mask_available = config.is_cdr_mask_available
    if is_cdr_mask_available:
        # cdr_mask_specs: list of tuples containing (column_name, output_filename, chain), or None if unavailable
        cdr_mask_specs = config.cdr_mask_specs
        if cdr_mask_specs is None:
            raise RuntimeError(
                "is_cdr_mask_available is True but cdr_mask_specs is None"
            )
        cdr_mask_specs_nn: list[tuple[str, str, str]] = cdr_mask_specs
        cdr_masks_by_column = {
            column_name: torch.zeros(dataset_size, max_length, dtype=torch.bool)
            for column_name, _, _ in cdr_mask_specs_nn
        }

    # Initialize paratop mask with zeros
    paratope_masks = torch.zeros(dataset_size, max_length, dtype=torch.bool)

    # Collect paratope and CDR mask values
    df = pd.read_csv(metadata_file)
    for row_idx, (_, row) in enumerate(df.iterrows()):
        vh_paratope_str = (
            "" if pd.isna(row[col_pa_vh_mask]) else str(row[col_pa_vh_mask])
        )
        vl_paratope_str = (
            "" if pd.isna(row[col_pa_vl_mask]) else str(row[col_pa_vl_mask])
        )

        combined_paratope_values = _extract_mask_values(
            vh_paratope_str + vl_paratope_str
        )
        paratope_tensor = torch.tensor(combined_paratope_values, dtype=torch.bool)
        paratope_masks[row_idx, : len(paratope_tensor)] = paratope_tensor

        if not is_cdr_mask_available:
            continue

        vl_start = 0 if pd.isna(row[col_vh_aa]) else len(row[col_vh_aa])

        for column_name, _, chain in cdr_mask_specs_nn:
            cdr_str = "" if pd.isna(row[column_name]) else str(row[column_name])
            cdr_values = _extract_mask_values(cdr_str)
            cdr_tensor = torch.tensor(cdr_values, dtype=torch.bool)

            start_idx = 0 if chain == "VH" else vl_start
            end_idx = start_idx + len(cdr_tensor)
            cdr_masks_by_column[column_name][row_idx, start_idx:end_idx] = cdr_tensor

    torch.save(paratope_masks, paratope_mask_file)

    if is_cdr_mask_available:
        for column_name, output_filename, _ in cdr_mask_specs_nn:
            torch.save(cdr_masks_by_column[column_name], output_dir / output_filename)


def generate_epitope_masks(config: PreprocessConfig) -> None:
    """Generate epitope masks for antigen sequences.

    Parameters
    ----------
    config: PreprocessConfig
        Configuration object containing preprocessing parameters.
    """
    metadata_file = config.metadata_file
    dataset_size = config.dataset_size
    max_length = config.length_ag
    col_ep_mask = config.col_ep_mask
    epitope_mask_file = config.epitope_mask_file

    # Initialize epitope mask tensor with zeros
    epitope_masks = torch.zeros(dataset_size, max_length, dtype=torch.bool)

    df = pd.read_csv(metadata_file)
    for row_idx, (_, row) in enumerate(df.iterrows()):
        epitope_str = "" if pd.isna(row[col_ep_mask]) else str(row[col_ep_mask])

        epitope = _extract_mask_values(epitope_str)
        epitope_tensor = torch.tensor(epitope, dtype=torch.bool)
        epitope_masks[row_idx, : len(epitope_tensor)] = epitope_tensor

    torch.save(epitope_masks, epitope_mask_file)


def sampling_by_group(
    df: pd.DataFrame,
    group_column: str,
    n_samples: int,
    random_seed: int = 12345,
    col_hash: str = "Unique_ag_vh_vl_hash",
) -> pd.DataFrame:
    """Sample rows by group with an upper cap per group.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to sample from.
    group_column : str
        Column used to define groups.
    n_samples : int
        Maximum number of rows to keep per group.
    random_seed : int, optional
        Random seed for deterministic sampling.
    col_hash : str, optional
        Column containing hash identifiers used to select rows.

    Returns
    -------
    pd.DataFrame
        Sampled DataFrame constrained by ``n_samples`` per group.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    sampled_df = df
    selected_hashes = set()
    for group_value, group_df in sampled_df.groupby(group_column):
        hashes = group_df[col_hash].tolist()
        if len(hashes) <= n_samples:
            selected_hashes.update(hashes)
        else:
            chosen = np.random.choice(hashes, n_samples, replace=False)
            selected_hashes.update(chosen.tolist())
            print(
                f"{group_column}={group_value} exceeds {n_samples} entries. Sampling down."
            )

    # return sampled_df[sampled_df[col_hash].isin(selected_hashes)]
    df_out = sampled_df[sampled_df[col_hash].isin(selected_hashes)]
    is_valid = validate_group_sampling(
        df_out,
        group_column=group_column,
        n_samples=n_samples,
        col_hash=col_hash,
    )
    if not is_valid:
        raise ValueError("Sampling validation failed.")

    return df_out


def validate_group_sampling(
    df: pd.DataFrame,
    group_column: str,
    n_samples: int,
    col_hash: str = "Unique_ag_vh_vl_hash",
) -> bool:
    """Return True if grouped sampling does not exceed the per-group cap.

    Parameters
    ----------
    df : pd.DataFrame
        Sampled dataframe.
    group_column : str
        Column used for grouped sampling (e.g., "cluster_id", "ab_id").
    n_samples : int
        Maximum number of rows allowed per group.
    col_hash : str, optional
        Column name for unique hash identifiers.
    """
    hash_list = df[col_hash].tolist()
    group_counts = df[df[col_hash].isin(hash_list)][group_column].value_counts()

    if (group_counts > n_samples).any():
        print(f"Warning: Some {group_column} groups exceed {n_samples} rows:")
        print(group_counts[group_counts > n_samples])
        return False

    print(f"All {group_column} groups have <= {n_samples} rows.")
    return True


def split_clusters_by_fold(
    cluster_ids: list[int],
    n_holdout: int,
    fold_idx: int,
) -> tuple[list[int], list[int]]:
    """Split ordered IDs into primary and holdout sets for a fold.

    Parameters
    ----------
    cluster_ids : list[int]
        Ordered cluster IDs to split.
    n_holdout : int
        Number of IDs in the holdout split.
    fold_idx : int
        Fold index used to locate holdout window.

    Returns
    -------
    tuple[list[int], list[int]]
        Tuple of ``(primary_ids, holdout_ids)``.
    """
    start_idx = fold_idx * n_holdout
    end_idx = start_idx + n_holdout
    if end_idx > len(cluster_ids):
        raise ValueError(
            f"Holdout split exceeds number of clusters. Maximum index is {len(cluster_ids) - 1}."
        )

    holdout_ids = cluster_ids[start_idx:end_idx]
    primary_ids = cluster_ids[:start_idx] + cluster_ids[end_idx:]

    return primary_ids, holdout_ids


def build_split_indices(
    df: pd.DataFrame,
    cluster_ids_split: dict[str, list[int]],
    col_cluster: str,
    col_hash: str,
    indices_file: Path,
) -> None:
    """Build and save split hash indices from cluster split definitions.

    Parameters
    ----------
    df : pd.DataFrame
        Metadata DataFrame.
    cluster_ids_split : dict
        Mapping from split name to cluster IDs.
    col_cluster : str
        Cluster column name in ``df``.
    col_hash : str
        Hash ID column name in ``df``.
    indices_file : Path
        Output JSON path for split hash indices.

    Returns
    -------
    None
        Writes validated split indices to ``indices_file``.
    """
    split_indices = {}
    for split_name, cluster_ids in cluster_ids_split.items():
        hashes = df.loc[df[col_cluster].isin(cluster_ids), col_hash].tolist()
        split_indices[split_name] = hashes

    is_valid = validate_splits(split_indices, cluster_ids_split)
    if not is_valid:
        raise ValueError("Split validation failed.")

    indices_file.parent.mkdir(parents=True, exist_ok=True)
    with open(indices_file, "w") as f:
        json.dump(split_indices, f)


def validate_splits(
    split_hashes: dict[str, list[str]],
    split_cluster_ids: dict[str, list[int]],
) -> bool:
    """Validate that there is no overlapping cluster ID or indices between phases.

    Parameters
    ----------
    split_hashes : dict[str, list[str]]
        Dictionary containing hash IDs per split.
    split_cluster_ids : dict[str, list[int]]
        Dictionary containing cluster IDs per split.

    Returns
    -------
    bool
        True if no overlap is found; otherwise False.
    """
    phases = list(split_hashes.keys())
    # Check for overlapping indices
    for i, phase1 in enumerate(phases):
        hashes1 = set(split_hashes[phase1])
        for phase2 in phases[i + 1 :]:
            hashes2 = set(split_hashes[phase2])
            overlap = hashes1 & hashes2
            if overlap:
                print(
                    f"Overlapping hashes between {phase1} and {phase2}: {sorted(overlap)[:10]} ... (total {len(overlap)})"
                )
                return False

    # Check for overlapping cluster IDs
    for i, phase1 in enumerate(phases):
        clusters1 = set(split_cluster_ids.get(phase1, []))
        for phase2 in phases[i + 1 :]:
            clusters2 = set(split_cluster_ids.get(phase2, []))
            cluster_overlap = clusters1 & clusters2
            if cluster_overlap:
                print(
                    f"Overlapping cluster_id between {phase1} and {phase2}: {sorted(cluster_overlap)[:10]} ... (total {len(cluster_overlap)})"
                )
                return False

    print("Validation passed: No overlapping indices or cluster IDs between phases.")
    return True


def load_hash_ids(file_hash_ids_list: list[str]) -> list[str]:
    """Load and merge hash IDs from multiple split-index JSON files.

    Parameters
    ----------
    file_hash_ids_list : list[str]
        List of JSON file paths containing split hash IDs.

    Returns
    -------
    list[str]
        De-duplicated merged hash IDs.
    """
    merged = []
    for file_hash_ids in file_hash_ids_list:
        with open(file_hash_ids) as f:
            split_indices = json.load(f)
        for _phase, hashes in split_indices.items():
            merged.extend(hashes)

    return list(set(merged))


def load_hash_to_indices(
    df: pd.DataFrame, hash_list: list[str], col_hash: str = "Unique_ag_vh_vl_hash"
) -> list[int]:
    """Load mapping from hash IDs to dataframe row indices.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the hash column.
    hash_list : list[str]
        List of hash IDs to map to row indices.
    col_hash : str
        Name of the column containing hash IDs.

    Returns
    -------
    list[int]
        List of row indices corresponding to the hash IDs.
    """
    hash_to_indices = [df.index[df[col_hash] == h][0] for h in hash_list]

    return hash_to_indices


def sample_and_save(file_in: Path, file_out: Path, hash_indices: list[int]) -> None:
    """Generalized function to sample and save either a torch tensor (.pt) or pandas dataframe (.csv).

    Parameters
    ----------
    file_in : Path
        Path to the input file (.pt or .csv).
    file_out : Path
        Path to save the sampled output file.
    hash_indices : list[int]
        List of row indices to sample.
    """
    input_suffix = file_in.suffix
    if input_suffix == ".pt":
        data = torch.load(file_in, weights_only=True)
        sampled = data[hash_indices]
        torch.save(sampled, file_out)
        print(f"Sampled tensor saved to {file_out} with shape {sampled.shape}")
    elif input_suffix == ".csv":
        df = pd.read_csv(file_in)
        df.reset_index(drop=True, inplace=True)

        sampled = df.iloc[hash_indices].reset_index(drop=True)
        sampled.to_csv(file_out, index=False)
        print(f"Sampled dataframe saved to {file_out} with {len(sampled)} rows.")
    else:
        raise ValueError(f"Unsupported file type: {file_in}")
