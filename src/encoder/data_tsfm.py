"""
Data loading for tSFM: memory-efficient index-based lookup.

Instead of pre-expanding all pairwise embeddings (20,540 rows × 512 × 1280
= ~54 GB for proteins alone), we store unique embeddings and look them up
at dataset __getitem__ time via index mappings.

Files expected in data_dir:
    ag_embed.pt     (N_unique_proteins, L_prot, 1280)  — unique TF protein embeddings
    ag_mask.pt      (N_unique_proteins, L_prot)         — protein padding masks
    ab_embed.pt     (N_unique_dna, L_dna, 768)          — unique DNA embeddings
    ab_mask.pt      (N_unique_dna, L_dna)                — DNA padding masks
    ag_indices.pt   (N_pairs,)                           — maps row → unique protein index
    ab_indices.pt   (N_pairs,)                           — maps row → unique DNA index
    metadata.csv    N_pairs rows with ag_id, ab_id, etc.
"""

from __future__ import annotations

import json
import os

import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset


class TSFMDataset(Dataset):
    """Dataset for TF–DNA pairs using index-based embedding lookup.

    Stores unique TF protein and DNA embeddings in memory,
    and maps each pairwise row to the correct embeddings at access time.
    This keeps memory at ~3 GB instead of ~55 GB.

    Optional per-pair source_idx tensor. When provided, __getitem__ returns
    an 8-tuple with source_idx[j] as the 8th element so the per-source
    temperature in CALMEncoder can be applied during training. When source_idx
    is None, returns the original 7-tuple unchanged; SFMs that do not use
    multi-source temperature default to source_idx=None and get byte-identical
    behavior.
    """

    def __init__(
        self,
        ag_embed: torch.Tensor,     # (N_unique_proteins, L_prot, 1280)
        ag_mask: torch.Tensor,      # (N_unique_proteins, L_prot)
        ab_embed: torch.Tensor,     # (N_unique_dna, L_dna, 768)
        ab_mask: torch.Tensor,      # (N_unique_dna, L_dna)
        ag_indices: torch.Tensor,   # (N_pairs,) index into ag_embed
        ab_indices: torch.Tensor,   # (N_pairs,) index into ab_embed
        ag_ids: torch.Tensor,       # (N_pairs,) TF identity IDs
        ab_ids: torch.Tensor,       # (N_pairs,) DNA identity IDs
        pair_indices: list[int],    # which rows to use in this split
        source_idx: torch.Tensor | None = None,  # (N_pairs,) per-pair source code
    ):
        self.ag_embed = ag_embed
        self.ag_mask = ag_mask
        self.ab_embed = ab_embed
        self.ab_mask = ab_mask
        self.ag_indices = ag_indices
        self.ab_indices = ab_indices
        self.ag_ids = ag_ids
        self.ab_ids = ab_ids
        self.pair_indices = torch.as_tensor(pair_indices, dtype=torch.long)
        self.source_idx = source_idx  # None for v1 SFMs (back-compat)

    def __len__(self) -> int:
        return int(self.pair_indices.numel())

    def __getitem__(self, idx: int):
        """Return per-pair tensors as a 7- or 8-tuple.

        Back-compat (source_idx is None): 7-tuple
            (ag, ab, ag_mask, ab_mask, ag_id, ab_id, j)
        v2 path (source_idx is a tensor): 8-tuple
            (ag, ab, ag_mask, ab_mask, ag_id, ab_id, j, source_idx[j])
        """
        j = int(self.pair_indices[idx])

        # Look up embeddings from unique pools
        ag_idx = int(self.ag_indices[j])
        ab_idx = int(self.ab_indices[j])

        ag = self.ag_embed[ag_idx]       # (L_prot, 1280)
        ab = self.ab_embed[ab_idx]       # (L_dna, 768)
        ag_mask = self.ag_mask[ag_idx]   # (L_prot,)
        ab_mask = self.ab_mask[ab_idx]   # (L_dna,)
        ag_id = self.ag_ids[j]
        ab_id = self.ab_ids[j]

        if self.source_idx is None:
            return ag, ab, ag_mask, ab_mask, ag_id, ab_id, j
        return ag, ab, ag_mask, ab_mask, ag_id, ab_id, j, self.source_idx[j]


def load_tsfm_data(cfg: DictConfig) -> dict:
    """Load tSFM data: unique embeddings + index mappings.

    Parameters
    ----------
    cfg : DictConfig
        Configuration with data.db pointing to preprocessed dir.

    Returns
    -------
    dict
        Dictionary with ag_embed, ag_mask, ab_embed, ab_mask,
        ag_indices, ab_indices, metadata.
    """
    db_cfg = cfg.data.db
    data_dir = db_cfg.data_dir

    # Domain-aware print labels (Phase 0.6, mhcSFM v2 audit cleanup).
    # Optional cfg.data.db.domain_labels lets each SFM provide its own labels.
    # Defaults preserve the v1 tSFM-origin behavior so existing SFM runs
    # produce identical log output and audits remain reproducible.
    domain_labels = getattr(db_cfg, "domain_labels", None) or {}
    sfm_label = domain_labels.get("sfm_name", "tSFM")
    agent_label = domain_labels.get("agent_label", "TF proteins")
    target_label = domain_labels.get("target_label", "DNA sequences")

    print(f"Loading {sfm_label} unique embeddings...")
    ag_embed = torch.load(os.path.join(data_dir, "ag_embed.pt"), weights_only=True)
    ag_mask = torch.load(os.path.join(data_dir, "ag_mask.pt"), weights_only=True)
    ab_embed = torch.load(os.path.join(data_dir, "ab_embed.pt"), weights_only=True)
    ab_mask = torch.load(os.path.join(data_dir, "ab_mask.pt"), weights_only=True)
    ag_indices = torch.load(os.path.join(data_dir, "ag_indices.pt"), weights_only=True)
    ab_indices = torch.load(os.path.join(data_dir, "ab_indices.pt"), weights_only=True)

    print(f"  {agent_label}: {ag_embed.shape[0]} unique, {ag_embed.shape}")
    print(f"  {target_label}: {ab_embed.shape[0]} unique, {ab_embed.shape}")
    print(f"  Total pairs: {len(ag_indices)}")

    metadata = pd.read_csv(os.path.join(data_dir, "metadata.csv"))

    # Per-source temperature (mhcSFM v2 Phase 1.5b): optionally load source_idx.pt
    # if present. v1 SFM data dirs don't have this file -> source_idx stays None
    # -> downstream behavior is byte-identical to v1.
    source_idx_path = os.path.join(data_dir, "source_idx.pt")
    if os.path.exists(source_idx_path):
        source_idx = torch.load(source_idx_path, weights_only=True)
        print(f"  source_idx: {source_idx.shape[0]} pairs, "
              f"unique sources: {sorted(set(source_idx.tolist()))}")
    else:
        source_idx = None

    return {
        "ag_embed": ag_embed,
        "ag_mask": ag_mask,
        "ab_embed": ab_embed,
        "ab_mask": ab_mask,
        "ag_indices": ag_indices,
        "ab_indices": ab_indices,
        "metadata": metadata,
        "source_idx": source_idx,
    }


def build_tsfm_dataloaders(
    cfg: DictConfig,
    split_phase: bool = True,
    merge_train_val: bool = False,
) -> dict[str, DataLoader]:
    """Build tSFM data loaders.

    Parameters
    ----------
    cfg : DictConfig
        Full configuration.
    split_phase : bool
        Whether to split into train/val/test.
    merge_train_val : bool
        Whether to merge train and validation sets.

    Returns
    -------
    dict[str, DataLoader]
        DataLoaders for each phase.
    """
    phaselist = cfg.phaselist
    num_workers = cfg.num_workers
    batch_size_train = cfg.train.encoder.batch_size
    batch_size_val = cfg.train.encoder.batch_size_val
    shuffle_train = cfg.train.encoder.shuffle_train

    data = load_tsfm_data(cfg)
    metadata = data["metadata"]

    # Load split indices from JSON
    file_split = cfg.paths.file_index_fold
    if not os.path.exists(file_split):
        raise FileNotFoundError(f"Split file not found: {file_split}")

    print(f"Loading split indices from {file_split}")
    with open(file_split) as f:
        hash_ids = json.load(f)

    # Map hashes to row indices
    metadata.reset_index(drop=True, inplace=True)
    hash_to_idx = {h: i for i, h in enumerate(metadata["Unique_ag_vh_vl_hash"])}

    split_indices = {}
    for phase_key, hashes in hash_ids.items():
        split_indices[phase_key] = [hash_to_idx[h] for h in hashes if h in hash_to_idx]

    if merge_train_val and "val" not in phaselist:
        split_indices["train"] = split_indices["train"] + split_indices.get("val", [])
        split_indices.pop("val", None)

    # Get IDs from metadata
    ag_ids = torch.as_tensor(metadata["ag_id"].values, dtype=torch.long)
    ab_ids = torch.as_tensor(metadata["ab_id"].values, dtype=torch.long)

    N_pairs = len(metadata)
    print(f"Loaded {N_pairs} total pairs.")
    for phase_key, indices in split_indices.items():
        print(f"  {phase_key}: {len(indices)} pairs")

    # Build datasets
    datasets = {}
    for phase in phaselist:
        phase_key = phase
        if phase_key not in split_indices:
            continue
        datasets[phase] = TSFMDataset(
            ag_embed=data["ag_embed"],
            ag_mask=data["ag_mask"],
            ab_embed=data["ab_embed"],
            ab_mask=data["ab_mask"],
            ag_indices=data["ag_indices"],
            ab_indices=data["ab_indices"],
            ag_ids=ag_ids,
            ab_ids=ab_ids,
            pair_indices=split_indices[phase_key],
            source_idx=data.get("source_idx"),  # None for v1 SFMs (back-compat)
        )

    batch_sizes = {
        "train": batch_size_train if isinstance(batch_size_train, int) else len(split_indices.get("train", [])),
        "val": batch_size_val,
        "test": batch_size_val,
    }

    dataloaders = {}
    for phase in phaselist:
        if phase not in datasets:
            continue
        dataloaders[phase] = DataLoader(
            datasets[phase],
            batch_size=batch_sizes.get(phase, batch_size_val),
            shuffle=(phase == "train" and shuffle_train),
            drop_last=(phase != "train"),
            num_workers=num_workers,
            pin_memory=True,
        )

    return dataloaders
