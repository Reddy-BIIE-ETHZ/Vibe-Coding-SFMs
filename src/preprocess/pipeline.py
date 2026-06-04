"""Preprocessing pipeline for CALM dataset construction, including embedding merging, mask generation, metadata processing, and data splitting."""

import random
from pathlib import Path

import pandas as pd
import torch

from calm.preprocess.cli import PreprocessConfig
from calm.preprocess.utils import (
    add_cluster_id_fields,
    add_epitope_paratope_fields,
    add_seq_id_fields,
    build_split_indices,
    generate_epitope_masks,
    generate_masks,
    generate_paratope_cdr_masks,
    load_hash_ids,
    load_hash_to_indices,
    sample_and_save,
    sampling_by_group,
    split_clusters_by_fold,
    update_metadata,
)


def prepare_model_inputs(config: PreprocessConfig, metadata: pd.DataFrame) -> None:
    """Prepare sampled splits and tensor files for model training.

    Parameters
    ----------
    config : PreprocessConfig
        Preprocessing configuration object.
    metadata : pd.DataFrame
        Full metadata table used to derive sampled subsets and output tensors.

    Returns
    -------
    None
        This function writes split indices and sampled tensor/csv outputs to disk.
    """
    samples_per_group = config.samples_per_group
    output_dir = config.output_dir
    output_indices_dir = config.output_indices_dir

    metadata_sampled = subsample_metadata(config, samples_per_group, metadata)
    build_data_splits(config, metadata_sampled, output_indices_dir)
    build_tensors(config, metadata, output_dir, output_indices_dir)


def merge_vh_vl(config: PreprocessConfig) -> None:
    """Concatenate VH and VL embeddings while removing internal padding.

    Parameters
    ----------
    config : PreprocessConfig
        Preprocessing configuration containing input/output embedding paths.

    Returns
    -------
    None
        Writes concatenated antibody embeddings to disk.
    """
    vh_embed_file = config.vh_embed_file
    vl_embed_file = config.vl_embed_file
    ab_embed_file = config.ab_embed_file

    vh_embed = torch.load(vh_embed_file, weights_only=True)
    vl_embed = torch.load(vl_embed_file, weights_only=True)

    batch_size, max_vh_len, embed_dim = vh_embed.shape
    _, max_vl_len, _ = vl_embed.shape

    # Detect valid (non-zero) positions in each embedding
    vh_mask = (vh_embed != 0).any(dim=-1)  # [B, L_vh]
    vl_mask = (vl_embed != 0).any(dim=-1)  # [B, L_vl]

    # Get actual lengths for each sequence
    vh_lengths = vh_mask.sum(dim=1)  # [B]
    vl_lengths = vl_mask.sum(dim=1)  # [B]

    # Create output tensor
    max_total_len = max_vh_len + max_vl_len
    ab_embed = torch.zeros(
        batch_size,
        max_total_len,
        embed_dim,
        dtype=vh_embed.dtype,
        device=vh_embed.device,
    )

    # Fill the concatenated embeddings for each sample in the batch
    for i in range(batch_size):
        vh_len = vh_lengths[i].item()
        vl_len = vl_lengths[i].item()

        # Copy valid VH tokens
        ab_embed[i, :vh_len] = vh_embed[i, :vh_len]

        # Copy valid VL tokens right after VH
        ab_embed[i, vh_len : vh_len + vl_len] = vl_embed[i, :vl_len]

    # Verify output tensor maintains the same batch order
    if ab_embed.shape[0] != batch_size:
        raise RuntimeError(
            f"Batch size changed during concatenation: {batch_size} -> {ab_embed.shape[0]}"
        )

    torch.save(ab_embed, ab_embed_file)
    print(f"Saved concatenated embeddings to {ab_embed_file}")


def build_masks(config: PreprocessConfig) -> None:
    """Build and save padding, epitope, paratope, and optional CDR masks.

    Parameters
    ----------
    config : PreprocessConfig
        Preprocessing configuration containing embedding and output mask paths.

    Returns
    -------
    None
        Writes generated mask tensors to disk.
    """
    ag_embed_file = config.ag_embed_file
    ab_embed_file = config.ab_embed_file
    ag_mask_file = config.ag_mask_file
    ab_mask_file = config.ab_mask_file

    # 1. Build Ag, Ab padding masks
    generate_masks(ag_embed_file, ag_mask_file)
    generate_masks(ab_embed_file, ab_mask_file)

    # 2. Build Epitope/Paratope masks
    generate_epitope_masks(config)
    generate_paratope_cdr_masks(config)


def build_metadata(config: PreprocessConfig) -> None:
    """Build normalized metadata with IDs and split-ready cluster fields.

    Parameters
    ----------
    config : PreprocessConfig
        Preprocessing configuration containing metadata paths and field names.

    Returns
    -------
    None
        This function writes processed metadata to disk.
    """
    input_metadata_file = config.input_metadata_file
    input_seq_file = config.input_seq_file
    metadata_file = config.metadata_file

    df = pd.read_csv(input_metadata_file)
    df_seq = pd.read_csv(input_seq_file)
    if len(df) != len(df_seq) or len(df) != config.dataset_size:
        raise ValueError(
            f"Metadata and sequence files have different number of samples: {len(df)} vs {len(df_seq)} (expected {config.dataset_size})"
        )

    # 1. Merge metadata and sequence dataframes
    col_intersect = set(df.columns).intersection(set(df_seq.columns))
    for col in col_intersect:
        identical = df[col].equals(df_seq[col])
        if not identical:
            df[col] = df_seq[col]
            print(f"Column {col} in metadata updated from sequence dataframe.")

    # 2. Create ag_id and ab_id
    df = add_seq_id_fields(df, config)

    # 3. Create epitope, paratope IDs and AA columns
    df = add_epitope_paratope_fields(df, config)

    df.reset_index(drop=True, inplace=True)
    df.to_csv(metadata_file, index=False)
    print(f"Metadata saved to {metadata_file} with {len(df)} rows.")


def load_metadata(config: PreprocessConfig) -> pd.DataFrame:
    """Load metadata and apply updates and cluster ID generation for splitting.

    Parameters
    ----------
    config : PreprocessConfig
        Preprocessing configuration containing metadata paths and cluster ID settings.

    Returns
    -------
    pd.DataFrame
        Processed metadata DataFrame ready for sampling and splitting.
    """
    df = pd.read_csv(config.metadata_file)

    # Update base metadata with src metadata if provided
    metadata_split_src_file = config.metadata_split_src_file
    if metadata_split_src_file is not None:
        df = update_metadata(df, config)

    # Create cluster IDs for train/test splitting
    df = add_cluster_id_fields(df, config)
    df.reset_index(drop=True, inplace=True)

    return df


def subsample_metadata(
    config: PreprocessConfig, n_samples: int, df: pd.DataFrame
) -> pd.DataFrame:
    """Subsample metadata in sequence by configured grouping columns.

    Parameters
    ----------
    config : PreprocessConfig
        Preprocessing configuration containing grouping column names and seed.
    n_samples : int
        Maximum rows retained per group at each sampling stage.
    df : pd.DataFrame
        Input metadata DataFrame.

    Returns
    -------
    pd.DataFrame
        Subsampled metadata DataFrame.
    """
    seed = config.seed
    col_group = [config.col_cluster_output, config.col_ab_id]
    col_hash = config.col_hash

    df_sampled = df
    for col in col_group:
        if col not in df.columns:
            raise KeyError(f"Column {col} not found in metadata for sampling.")

        df_sampled = sampling_by_group(
            df_sampled,
            group_column=col,
            n_samples=n_samples,
            random_seed=seed,
            col_hash=col_hash,
        )

    return df_sampled


def build_data_splits(
    config: PreprocessConfig, df: pd.DataFrame, output_dir: Path
) -> None:
    """Create nested CV split index files based on cluster IDs.

    Parameters
    ----------
    config : PreprocessConfig
        Preprocessing configuration containing split settings and column names.
    df : pd.DataFrame
        Metadata DataFrame with cluster IDs.
    output_dir : Path
        Directory where split hash-index JSON files are written.

    Returns
    -------
    None
        Writes split index files for each outer/inner fold.
    """
    col_cluster = config.col_cluster_output
    col_hash = config.col_hash
    seed = config.seed

    cv_folds = config.cv_folds
    test_ratio = 1 / cv_folds
    val_ratio = 1 / cv_folds

    cluster_ids = df[col_cluster].unique().tolist()
    random.seed(seed)
    random.shuffle(cluster_ids)

    n_clusters = len(cluster_ids)
    n_clusters_test = int(n_clusters * test_ratio)
    n_clusters_val = int(n_clusters * val_ratio)

    for outer_fold in range(cv_folds):
        print(f"Processing outer fold {outer_fold}...")

        # 1. Split ref testset
        cluster_ids_trainval, cluster_ids_test = split_clusters_by_fold(
            cluster_ids, n_clusters_test, outer_fold
        )

        # 2. Split trainval into train and val with predefined testset
        random.shuffle(cluster_ids_trainval)
        n_clusters_val = int(len(cluster_ids_trainval) * val_ratio)

        for inner_fold in range(cv_folds):
            indices_file = (
                output_dir
                / f"split_hash_ids_outerfold_{outer_fold}_innerfold_{inner_fold}.json"
            )

            cluster_ids_train, cluster_ids_val = split_clusters_by_fold(
                cluster_ids_trainval, n_clusters_val, inner_fold
            )

            cluster_ids_split = {
                "train": cluster_ids_train,
                "val": cluster_ids_val,
                "test": cluster_ids_test,
            }

            build_split_indices(
                df,
                cluster_ids_split,
                col_cluster=col_cluster,
                col_hash=col_hash,
                indices_file=indices_file,
            )


def build_tensors(
    config: PreprocessConfig,
    metadata: pd.DataFrame,
    output_dir: Path,
    output_indices_dir: Path,
) -> None:
    """Materialize sampled tensors and metadata from split hash-index files.

    Parameters
    ----------
    config : PreprocessConfig
        Preprocessing configuration containing source tensor paths.
    metadata : pd.DataFrame
        Full metadata DataFrame to be sampled.
    output_dir : Path
        Destination directory for sampled outputs.
    output_indices_dir : Path
        Directory containing split hash-index JSON files.

    Returns
    -------
    None
        Writes sampled metadata and tensor files to ``output_dir``.
    """
    file_hash_ids = [str(p) for p in sorted(output_indices_dir.glob("*.json"))]
    hash_ids = load_hash_ids(file_hash_ids)

    # hash id to row indices mapping
    col_hash = config.col_hash
    hash_to_indices = load_hash_to_indices(metadata, hash_ids, col_hash=col_hash)

    # Save metadata
    metadata_file_output = output_dir / "metadata.csv"
    metadata.reset_index(drop=True, inplace=True)
    metadata_out = metadata.iloc[hash_to_indices].reset_index(drop=True)
    metadata_out.to_csv(metadata_file_output, index=False)
    print(f"Metadata saved to {metadata_file_output} with {len(metadata_out)} rows.")

    # Save Ag embedding
    ag_embed_file_input = config.ag_embed_file
    ag_embed_file_output = output_dir / "ag_embed.pt"
    sample_and_save(ag_embed_file_input, ag_embed_file_output, hash_to_indices)

    # Save Ab embedding
    ab_embed_file_input = config.ab_embed_file
    ab_embed_file_output = output_dir / ab_embed_file_input.name
    sample_and_save(ab_embed_file_input, ab_embed_file_output, hash_to_indices)

    # Save masks
    ag_mask_file_input = config.ag_mask_file
    ag_mask_file_output = output_dir / ag_mask_file_input.name
    sample_and_save(ag_mask_file_input, ag_mask_file_output, hash_to_indices)

    ab_mask_file_input = config.ab_mask_file
    ab_mask_file_output = output_dir / ab_mask_file_input.name
    sample_and_save(ab_mask_file_input, ab_mask_file_output, hash_to_indices)

    epitope_mask_file_input = config.epitope_mask_file
    epitope_mask_file_output = output_dir / epitope_mask_file_input.name
    sample_and_save(epitope_mask_file_input, epitope_mask_file_output, hash_to_indices)

    paratope_mask_file_input = config.paratope_mask_file
    paratope_mask_file_output = output_dir / paratope_mask_file_input.name
    sample_and_save(
        paratope_mask_file_input, paratope_mask_file_output, hash_to_indices
    )

    if config.is_cdr_mask_available:
        cdr_mask_specs = config.cdr_mask_specs
        if cdr_mask_specs is None:
            raise RuntimeError(
                "is_cdr_mask_available is True but cdr_mask_specs is None"
            )
        for _column_name, cdr_mask_file_input_name, _ in cdr_mask_specs:
            cdr_mask_file_input = config.input_dir / cdr_mask_file_input_name
            cdr_mask_file_output = output_dir / cdr_mask_file_input_name
            sample_and_save(cdr_mask_file_input, cdr_mask_file_output, hash_to_indices)
