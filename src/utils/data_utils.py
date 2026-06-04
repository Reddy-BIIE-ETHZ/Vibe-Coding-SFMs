"""Utility functions for data loading and splitting in the CALM pipeline."""

from __future__ import annotations

import json
import os
import random

import pandas as pd
import torch
from omegaconf import DictConfig


def load_embeddings(
    cfg: DictConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load antigen and antibody embeddings from files.

    Parameters
    ----------
    cfg : DictConfig
        The arguments containing data loading parameters.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple containing (ag_embed, ab_embed).
    """
    ag_embed = torch.load(cfg.file_ag, weights_only=True)
    ab_embed = torch.load(cfg.file_ab, weights_only=True)

    # Validate shapes
    if ag_embed.shape[0] != ab_embed.shape[0]:
        raise ValueError(
            f"Number of antigen embeddings ({ag_embed.shape[0]}) does not match number of antibody embeddings ({ab_embed.shape[0]})."
        )
    return ag_embed, ab_embed


def load_masks(cfg: DictConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Load antigen and antibody masks from files.

    Parameters
    ----------
    cfg : DictConfig
        The arguments containing data loading parameters.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple containing (ag_mask, ab_mask).
    """
    ag_mask = torch.load(cfg.file_agmask, weights_only=True)
    ab_mask = torch.load(cfg.file_abmask, weights_only=True)

    # Validate shapes
    if ag_mask.shape[0] != ab_mask.shape[0]:
        raise ValueError(
            f"Number of antigen masks ({ag_mask.shape[0]}) does not match number of antibody masks ({ab_mask.shape[0]})."
        )
    return ag_mask, ab_mask


def apply_masks(
    cfg: DictConfig,
    embeddings: torch.Tensor,
    padding_masks: torch.Tensor,
    mask: str = "epitope",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply specified masks to the embeddings.

    Parameters
    ----------
    cfg : DictConfig
        The data configuration containing mask file paths.
    embeddings : torch.Tensor
        The original embeddings.
    padding_masks : torch.Tensor
        The padding masks for the embeddings.
    mask : str, optional
        The type of mask to apply ('epitope', 'paratope', or 'cdr'), by default "epitope"

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple containing:
            - masked_embeddings: Embeddings with the specified mask applied.
            - padding_masks_updated: Updated padding masks after intersection with the applied mask.
    """
    if mask == "epitope":
        mask_tensor = torch.load(cfg.file_epitope_mask, weights_only=True)
    elif mask == "paratope":
        mask_tensor = torch.load(cfg.file_paratope_mask, weights_only=True)
    elif mask == "cdr":
        cdr_mask_dir = cfg.data_dir
        cdr_mask_files = cfg.file_cdr_mask
        mask_tensor = None
        for _key, file in cdr_mask_files.items():
            loaded_mask = torch.load(
                os.path.join(cdr_mask_dir, file), weights_only=True
            )
            if mask_tensor is None:
                mask_tensor = loaded_mask.bool()
            else:
                mask_tensor = mask_tensor | loaded_mask.bool()
        mask_tensor = mask_tensor.float()
    else:
        raise ValueError(f"Unknown mask type: {mask}")

    # Expand mask to match embedding dimensions
    mask_expanded = mask_tensor.unsqueeze(-1).expand_as(embeddings)
    masked_embeddings = embeddings * mask_expanded.float()

    padding_masks_updated = padding_masks & mask_tensor.bool()

    return masked_embeddings, padding_masks_updated


def load_metadata(cfg: DictConfig) -> pd.DataFrame:
    """Load the metadata CSV file from the specified directory.

    Parameters
    ----------
    cfg : DictConfig
        The arguments containing data loading parameters.

    Returns
    -------
    df : pd.DataFrame
        The loaded metadata as a pandas DataFrame.
    """
    df = pd.read_csv(cfg.file_metadata)
    return df


def load_sequences(
    db_cfg: DictConfig, loader_cfg: DictConfig
) -> tuple[list[str], list[str]]:
    """Load antigen and antibody sequences from the specified files.

    Parameters
    ----------
    db_cfg : DictConfig
        The database configuration containing file paths.
    loader_cfg : DictConfig
        The loader configuration containing column names.

    Returns
    -------
    tuple[list[str], list[str]]
        A tuple containing (ag_seq, ab_seq).
    """
    df = pd.read_csv(db_cfg.file_metadata)

    ag_seq = df[loader_cfg.col_seq_ag].tolist()

    if hasattr(loader_cfg, "col_seq_vh") and hasattr(loader_cfg, "col_seq_vl"):
        ab_seq = [
            str(vh) + str(vl)
            for vh, vl in zip(
                df[loader_cfg.col_seq_vh].tolist(),
                df[loader_cfg.col_seq_vl].tolist(),
                strict=False,
            )
        ]
    elif hasattr(loader_cfg, "col_seq_ab"):
        ab_seq = df[loader_cfg.col_seq_ab].tolist()
    else:
        raise ValueError(
            "Antibody sequences not found. Please specify either 'col_seq_ab' or both 'col_seq_vh' and 'col_seq_vl' in the configuration."
        )

    return ag_seq, ab_seq


def load_seq_ids(
    db_cfg: DictConfig, loader_cfg: DictConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load antigen and antibody sequence IDs from the specified files.

    Parameters
    ----------
    db_cfg : DictConfig
        The database configuration containing file paths.
    loader_cfg : DictConfig
        The loader configuration containing column names.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple containing (ag_ids, ab_ids) as long tensors.
    """
    df = pd.read_csv(db_cfg.file_metadata)

    ag_ids = torch.as_tensor(df[loader_cfg.col_ag_id].values, dtype=torch.long)
    ab_ids = torch.as_tensor(df[loader_cfg.col_ab_id].values, dtype=torch.long)
    return ag_ids, ab_ids


def load_split_indices(file_indices: str, file_df: str) -> dict[str, list[int]]:
    """Load pre-saved split indices from a JSON file.

    Parameters
    ----------
    file_indices : str
        The path to the JSON file containing split hash IDs.
    file_df : str
        The path to the metadata CSV file used to map hashes to row indices.

    Returns
    -------
    dict[str, list[int]]
        A dictionary containing row indices for each phase.
            - 'train': List of training indices
            - 'val': List of validation indices
            - 'test': List of test indices
    """
    df = pd.read_csv(file_df)
    df.reset_index(drop=True, inplace=True)

    with open(file_indices) as f:
        hash_ids = json.load(f)

    split_indices = {}
    for phase, hashes in hash_ids.items():
        indices = [int(df.index[df["Unique_ag_vh_vl_hash"] == h][0]) for h in hashes]
        split_indices[phase] = indices

    return split_indices


def load_predefined_cluster_indices(
    file_predefined_clusters: str,
) -> dict[str, list[int]] | None:
    """Load pre-saved cluster indices for splitting from a JSON file.

    Parameters
    ----------
    file_predefined_clusters : str
        The path to the JSON file containing pre-defined cluster indices.

    Returns
    -------
    cluster_indices : dict | None
        A dictionary containing indices for each phase.
            - 'train': List of pre-defined training cluster indices
            - 'val': List of pre-defined validation cluster indices
            - 'test': List of pre-defined test cluster indices
    """
    # file_predefined_clusters = cfg.file_predefined_clusters
    if file_predefined_clusters is None or not os.path.exists(file_predefined_clusters):
        return None
    else:
        with open(file_predefined_clusters) as f:
            cluster_indices: dict[str, list[int]] = json.load(f)

        return cluster_indices


def load_cluster_id(
    cfg: DictConfig, indices: list[int] | None = None
) -> list[int | str]:
    """Load cluster IDs from the metadata file.

    Parameters
    ----------
    cfg : DictConfig
        The configuration containing data loading parameters.
    indices : list[int], optional
        Row indices to restrict loading to, by default None (loads all rows).

    Returns
    -------
    list[int | str]
        A list of cluster IDs per row.
    """
    data_cfg = cfg.data.db
    col_id = data_cfg.col_cluster_id

    df = load_metadata(data_cfg)
    if indices is not None:
        df = df.iloc[indices]
    cluster_ids: list[int | str] = df[col_id].tolist()
    return cluster_ids


def load_per_cluster_indices(
    cfg: DictConfig, split_indices: dict[str, list[int]]
) -> dict[str, list[int]]:
    """Return one representative row index per cluster for each split phase.

    Parameters
    ----------
    cfg : DictConfig
        The configuration containing data loading parameters.
    split_indices : dict[str, list[int]]
        A dictionary mapping phase names to lists of row indices.

    Returns
    -------
    dict[str, list[int]]
        A dictionary mapping phase names to a list of one row index per unique
        cluster ID found in that phase.
    """
    result: dict[str, list[int]] = {}
    for phase, indices in split_indices.items():
        cluster_ids = load_cluster_id(cfg, indices)
        seen: set[int | str] = set()
        representative: list[int] = []
        for row_idx, cluster_id in zip(indices, cluster_ids, strict=False):
            if cluster_id not in seen:
                seen.add(cluster_id)
                representative.append(row_idx)
        result[phase] = representative
    return result


def reduce_masked_embeddings(
    embeddings: torch.Tensor, masks: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce embeddings by selecting only the valid (unmasked) positions.

    Parameters
    ----------
    embeddings : torch.Tensor
        The input embeddings of shape (batch_size, seq_len, embed_dim).
    masks : torch.Tensor
        The attention masks of shape (batch_size, seq_len), where 1 indicates valid positions and 0 indicates masked positions.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple containing:
            - reduced_embeddings: The reduced embeddings tensor.
            - reduced_masks: The corresponding reduced masks tensor.
    """
    reduced_embeddings: list[torch.Tensor] = []
    reduced_masks: list[torch.Tensor] = []

    # If masks is 3D, squeeze to 2D (batch_size, seq_len)
    if masks.dim() == 3:
        masks = masks.squeeze(-1)

    for emb, mask in zip(embeddings, masks, strict=False):
        # emb: (seq_len, embed_dim), mask: (seq_len,)
        new_emb_items: list[torch.Tensor] = []
        new_mask_items: list[bool] = []
        prev_false = False
        zero_vec = torch.zeros_like(emb[0])
        for i in range(len(mask)):
            if mask[i]:
                new_mask_items.append(True)
                new_emb_items.append(emb[i])
                prev_false = False
            else:
                if not prev_false:
                    new_mask_items.append(False)
                    new_emb_items.append(zero_vec)
                    prev_false = True
                # else: skip consecutive False
        new_emb: torch.Tensor = (
            torch.stack(new_emb_items, dim=0) if new_emb_items else emb[:0]
        )
        new_mask: torch.Tensor = torch.tensor(
            new_mask_items, dtype=mask.dtype, device=mask.device
        )
        reduced_embeddings.append(new_emb)
        reduced_masks.append(new_mask)

    # Pad reduced_embeddings and reduced_masks to the same length for batching
    max_len = max(e.shape[0] for e in reduced_embeddings)
    embed_dim = reduced_embeddings[0].shape[1] if reduced_embeddings else 0
    batch_size = len(reduced_embeddings)
    device = embeddings.device if isinstance(embeddings, torch.Tensor) else "cpu"
    dtype = embeddings.dtype if isinstance(embeddings, torch.Tensor) else torch.float32
    mask_dtype = masks.dtype if isinstance(masks, torch.Tensor) else torch.bool

    padded_embeddings = torch.zeros(
        (batch_size, max_len, embed_dim), dtype=dtype, device=device
    )
    padded_masks = torch.zeros((batch_size, max_len), dtype=mask_dtype, device=device)
    for i, (emb, msk) in enumerate(
        zip(reduced_embeddings, reduced_masks, strict=False)
    ):
        padded_embeddings[i, : emb.shape[0], :] = emb
        padded_masks[i, : msk.shape[0]] = msk

    if masks.dim() == 3:
        padded_masks = padded_masks.unsqueeze(-1)

    return padded_embeddings, padded_masks


def decluster_splits(
    cfg: DictConfig, split_indices: dict[str, list[int]]
) -> dict[str, list[int]]:
    """Adjust clustered splits to decluster while not including NNK samples in val/test set.

    Parameters
    ----------
    cfg : DictConfig
        The arguments passed from the command line.
    split_indices : dict[str, list[int]]
        A dictionary containing split indices for each phase.

    Returns
    -------
    dict[str, list[int]]
        The adjusted split indices without NNK samples.
    """
    data_cfg = cfg.data.db
    random_seed = cfg.random_seed
    random.seed(random_seed)

    df = load_metadata(data_cfg)

    all_indices = split_indices["train"] + split_indices["val"] + split_indices["test"]
    all_indices_non_nnk = [
        indices
        for indices in all_indices
        if "NNK" not in str(df.iloc[indices]["Data_Source"])
    ]

    # Redistribute indices randomly into train, val, and test splits
    train_size = len(split_indices["train"])
    val_size = len(split_indices["val"])
    test_size = len(split_indices["test"])

    if len(all_indices_non_nnk) < val_size + test_size:
        raise ValueError(
            f"Not enough non-NNK samples ({len(all_indices_non_nnk)}) to fill the requested splits (train: {train_size}, val: {val_size}, test: {test_size})."
        )

    test_indices_declustered = (
        random.sample(all_indices_non_nnk, test_size) if test_size > 0 else []
    )
    val_indices_declustered = (
        random.sample(
            [idx for idx in all_indices_non_nnk if idx not in test_indices_declustered],
            val_size,
        )
        if val_size > 0
        else []
    )
    train_indices_declustered = [
        idx
        for idx in all_indices
        if idx not in (test_indices_declustered + val_indices_declustered)
    ]

    split_indices_declustered = {
        "train": train_indices_declustered,
        "val": val_indices_declustered,
        "test": test_indices_declustered,
    }

    if len(train_indices_declustered) != train_size:
        raise ValueError(
            f"Train split size mismatch after declustering: expected {train_size}, got {len(train_indices_declustered)}."
        )
    if len(val_indices_declustered) != val_size:
        raise ValueError(
            f"Validation split size mismatch after declustering: expected {val_size}, got {len(val_indices_declustered)}."
        )
    if len(test_indices_declustered) != test_size:
        raise ValueError(
            f"Test split size mismatch after declustering: expected {test_size}, got {len(test_indices_declustered)}."
        )

    return split_indices_declustered


def shuffle_ag_ab_pairs(
    split_indices: dict[str, list[int]],
    ag_embed: torch.Tensor,
    ag_mask: torch.Tensor,
    ab_embed: torch.Tensor,
    ab_mask: torch.Tensor,
    ag_ids: torch.Tensor,
    ab_ids: torch.Tensor,
    random_seed: int = 12345,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Shuffle antigen and antibody pairs in the training set for baseline experiments.

    Parameters
    ----------
    split_indices : dict
        A dictionary containing split indices for each phase.
    ag_embed : torch.Tensor
        Antigen embeddings.
    ag_mask : torch.Tensor
        Antigen masks.
    ab_embed : torch.Tensor
        Antibody embeddings.
    ab_mask : torch.Tensor
        Antibody masks.
    ag_ids : torch.Tensor
        Antigen sequence IDs.
    ab_ids : torch.Tensor
        Antibody sequence IDs.
    random_seed : int, optional
        Random seed for shuffling, by default 12345.

    Returns
    -------
    tuple
        Shuffled antigen and antibody embeddings, masks, and sequence IDs.
    """
    # Ag shuffling
    random_split_indices_ag = split_indices["train"].copy()
    random.seed(random_seed)
    random.shuffle(random_split_indices_ag)

    ag_embed[split_indices["train"]] = ag_embed[random_split_indices_ag]
    ag_mask[split_indices["train"]] = ag_mask[random_split_indices_ag]
    ag_ids_shuffled = ag_ids.clone()
    ag_ids_shuffled[split_indices["train"]] = ag_ids[random_split_indices_ag]
    ag_ids = ag_ids_shuffled

    # Ab shuffling
    random_split_indices_ab = split_indices["train"].copy()
    random.seed(random_seed + 1)
    random.shuffle(random_split_indices_ab)

    ab_embed[split_indices["train"]] = ab_embed[random_split_indices_ab]
    ab_mask[split_indices["train"]] = ab_mask[random_split_indices_ab]
    ab_ids_shuffled = ab_ids.clone()
    ab_ids_shuffled[split_indices["train"]] = ab_ids[random_split_indices_ab]
    ab_ids = ab_ids_shuffled

    # Ensure indices are within bounds
    if max(random_split_indices_ab) >= len(ab_ids):
        raise IndexError("Shuffled antibody indices exceed the length of ab_ids.")
    if max(random_split_indices_ag) >= len(ag_ids):
        raise IndexError("Shuffled antigen indices exceed the length of ag_ids.")

    return ag_embed, ag_mask, ab_embed, ab_mask, ag_ids, ab_ids
