"""
Data loading and preprocessing.

Includes PyTorch DataLoader and functions to load embeddings and metadata, split data into
"""

from __future__ import annotations

import os
import random
from collections import defaultdict
from collections.abc import Iterator
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, Sampler

from calm.utils.data_utils import (
    apply_masks,
    decluster_splits,
    load_cluster_id,
    load_embeddings,
    load_masks,
    load_seq_ids,
    load_split_indices,
    reduce_masked_embeddings,
    shuffle_ag_ab_pairs,
)


def load_encoder_inputs(
    cfg: DictConfig,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Load encoder inputs including embeddings, masks, and sequence IDs.

    Parameters
    ----------
    cfg : DictConfig
        The arguments containing data loading parameters.

    Returns
    -------
    tuple[ torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[str] ]
        A tuple containing the following elements:
            - ag_embed: Antigen embeddings
            - ab_embed: Antibody embeddings
            - ag_mask: Antigen attention masks
            - ab_mask: Antibody attention masks
            - ag_ids: Antigen sequence IDs
            - ab_ids: Antibody sequence IDs
    """
    db_cfg = cfg.data.db
    loader_cfg = cfg.data.loader

    use_epitope_mask = loader_cfg.use_epitope_mask
    use_paratope_mask = loader_cfg.use_paratope_mask
    use_cdr_mask = loader_cfg.use_cdr_mask
    reduce_embeddings = cfg.train.encoder.reduce_embeddings

    ag_embed, ab_embed = load_embeddings(db_cfg)
    ag_mask, ab_mask = load_masks(db_cfg)
    ag_ids, ab_ids = load_seq_ids(db_cfg, loader_cfg)

    if ag_embed.shape[0] != ag_mask.shape[0] or ag_embed.shape[0] != ag_ids.shape[0]:
        raise ValueError(
            f"Antigen data size mismatch: ag_embed:{ag_embed.shape[0]}, ag_mask:{ag_mask.shape[0]}, ag_ids:{ag_ids.shape[0]}"
        )
    if ab_embed.shape[0] != ab_mask.shape[0] or ab_embed.shape[0] != ab_ids.shape[0]:
        raise ValueError(
            f"Antibody data size mismatch: ab_embed:{ab_embed.shape[0]}, ab_mask:{ab_mask.shape[0]}, ab_ids:{ab_ids.shape[0]}"
        )

    if use_epitope_mask:
        ag_embed, ag_mask = apply_masks(db_cfg, ag_embed, ag_mask, mask="epitope")
    if use_paratope_mask:
        ab_embed, ab_mask = apply_masks(db_cfg, ab_embed, ab_mask, mask="paratope")
    if use_cdr_mask:
        ab_embed, ab_mask = apply_masks(db_cfg, ab_embed, ab_mask, mask="cdr")

    if reduce_embeddings:
        ag_embed, ag_mask = reduce_masked_embeddings(ag_embed, ag_mask)
        ab_embed, ab_mask = reduce_masked_embeddings(ab_embed, ab_mask)

    return (
        ag_embed,
        ab_embed,
        ag_mask,
        ab_mask,
        ag_ids,
        ab_ids,
    )


def build_encoder_dataloaders(
    cfg: DictConfig, split_phase: bool = True, merge_train_val: bool = False
) -> dict[
    str,
    DataLoader[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            int,
        ]
    ],
]:
    """Build data loaders for training, validation, and test sets.

    Parameters
    ----------
    cfg : DictConfig
        The arguments containing data loading parameters.
    split_phase : bool
        Whether to split the data into phases (train/val/test).
    merge_train_val : bool, optional
        Whether to merge training and validation datasets, by default False.

    Returns
    -------
    dataloaders : dict
        A dictionary containing PyTorch DataLoader objects for each phase.
            - 'train': DataLoader for training set
            - 'val': DataLoader for validation set
            - 'test': DataLoader for test set
    """
    # Global cfg
    num_workers = cfg.num_workers
    phaselist = cfg.phaselist
    random_seed = cfg.random_seed
    file_split = cfg.paths.file_index_fold

    # Train cfg
    batch_size_train = cfg.train.encoder.batch_size
    batch_size_val = cfg.train.encoder.batch_size_val
    shuffle_train = cfg.train.encoder.shuffle_train
    use_clusterwise_sampler = cfg.train.encoder.use_clusterwise_sampler
    clusterwise_sampler_k = cfg.train.encoder.clusterwise_sampler_k
    baseline_decluster_splits = cfg.train.encoder.baseline_decluster_splits
    baseline_shuffle_pairs = cfg.train.encoder.baseline_shuffle_pairs

    # Load data
    ag_embed, ab_embed, ag_mask, ab_mask, ag_ids, ab_ids = load_encoder_inputs(cfg)
    print(f"Loaded {ag_embed.shape[0]} samples.")

    if split_phase:
        # Get data splits
        if os.path.exists(file_split):
            file_metadata = cfg.data.db.file_metadata
            print(f"Using existing split indices from {file_split}")
            split_indices = load_split_indices(file_split, file_metadata)
        else:
            raise FileNotFoundError(f"Split indices file not found: {file_split}")

        # Adjust splits for baseline checks
        if baseline_decluster_splits:
            print("Applying declustering to data splits for baseline experiment.")
            split_indices = decluster_splits(cfg, split_indices)

        if merge_train_val and ("val" not in phaselist):
            print("Merging training and validation datasets for final training.")
            split_indices["train"] = split_indices["train"] + split_indices["val"]
            del split_indices["val"]

        if baseline_shuffle_pairs:
            print(
                "Applying random shuffling of antigen-antibody pairs for baseline experiment."
            )
            ag_embed, ag_mask, ab_embed, ab_mask, ag_ids, ab_ids = shuffle_ag_ab_pairs(
                split_indices,
                ag_embed,
                ag_mask,
                ab_embed,
                ab_mask,
                ag_ids,
                ab_ids,
                random_seed=random_seed,
            )

        datasets = {
            phase: AgAbPairDataset(
                ag_embed,
                ab_embed,
                ag_mask,
                ab_mask,
                ag_ids,
                ab_ids,
                split_indices[phase],
            )
            for phase in phaselist
        }

        # Controlling batch sizes for different phases
        batch_sizes = {
            "train": (
                batch_size_train
                if isinstance(batch_size_train, int)
                else len(split_indices["train"])
            ),
            "val": batch_size_val,
            "test": batch_size_val,
        }

        if use_clusterwise_sampler:
            cluster_id_train = load_cluster_id(cfg, indices=split_indices["train"])
            sampler = ClusterRotatingSampler(cluster_id_train, k=clusterwise_sampler_k)
            batch_sizes["train"] = len(sampler)

            dataloaders = {
                phase: DataLoader(
                    datasets[phase],
                    batch_size=batch_sizes[phase],
                    sampler=(sampler if phase == "train" else None),
                    shuffle=False,
                    drop_last=True,
                    num_workers=num_workers,
                    pin_memory=True,
                )
                for phase in phaselist
            }
        else:
            dataloaders = {
                phase: DataLoader(
                    datasets[phase],
                    batch_size=batch_sizes[phase],
                    shuffle=(phase == "train" and shuffle_train),
                    drop_last=(phase != "train"),
                    num_workers=num_workers,
                    pin_memory=True,
                )
                for phase in phaselist
            }
    else:
        # Use the entire dataset without splitting
        dataset = AgAbPairDataset(
            ag_embed,
            ab_embed,
            ag_mask,
            ab_mask,
            ag_ids,
            ab_ids,
            list(range(ag_embed.shape[0])),
        )

        dataloaders = {
            phase: DataLoader(
                dataset,
                batch_size=batch_size_train,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers,
                pin_memory=True,
            )
            for phase in phaselist
        }

    return dataloaders


class ClusterRotatingSampler(Sampler[int]):
    """Samples up to k data points per cluster per epoch.

    If a cluster has fewer than k data points, sample all available.

    Parameters
    ----------
    cluster_id : list[Any]
        List or array of cluster IDs for each data point.
    k : int, optional
        Number of samples per cluster per epoch, by default 1.
    shuffle_within_cluster : bool, optional
        Whether to shuffle indices within each cluster initially, by default True.
    shuffle_clusters : bool, optional
        Whether to shuffle the order of clusters each epoch, by default True.
    """

    def __init__(
        self,
        cluster_id: list[Any],
        k: int = 1,
        shuffle_within_cluster: bool = True,
        shuffle_clusters: bool = True,
    ) -> None:
        """Initialize the ClusterRotatingSampler.

        Parameters
        ----------
        cluster_id : list[Any]
            List or array of cluster IDs for each data point.
        k : int, optional
            Number of samples per cluster per epoch, by default 1.
        shuffle_within_cluster : bool, optional
            Whether to shuffle indices within each cluster initially, by default True.
        shuffle_clusters : bool, optional
            Whether to shuffle the order of clusters each epoch, by default True.
        """
        self.cluster_id = cluster_id
        self.k = k
        self.shuffle_clusters = shuffle_clusters
        self.rng = random.SystemRandom()

        # group indices by cluster
        self.by_cluster = defaultdict(list)
        for idx, c in enumerate(self.cluster_id):
            self.by_cluster[c].append(idx)

        # shuffle within cluster once initially
        if shuffle_within_cluster:
            for c in self.by_cluster:
                self.rng.shuffle(self.by_cluster[c])

        # store cluster list
        self.clusters = sorted(self.by_cluster.keys())

        # track position within each cluster
        self.ptr = {c: 0 for c in self.clusters}

    def __iter__(self) -> Iterator[int]:
        """Iterate over the dataset, yielding indices."""
        clusters = self.clusters[:]
        if self.shuffle_clusters:
            self.rng.shuffle(clusters)

        selected = []
        for c in clusters:
            idxs = self.by_cluster[c]
            n = len(idxs)
            k = min(self.k, n)
            pos = self.ptr[c]
            # select k indices, wrap around if needed
            for i in range(k):
                selected.append(idxs[(pos + i) % n])
            # move pointer forward cyclically
            self.ptr[c] = (pos + k) % n
        return iter(selected)

    def __len__(self) -> int:
        """Get the total number of samples per epoch."""
        # total number of samples per epoch
        return sum(min(self.k, len(self.by_cluster[c])) for c in self.clusters)


class AgAbPairDataset(
    Dataset[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            int,
        ]
    ]
):
    """Dataset for antigen-antibody pairs with in-batch negatives.

    Parameters
    ----------
    Dataset : torch.utils.data.Dataset
        PyTorch Dataset class
    """

    def __init__(
        self,
        ag_emb: torch.Tensor,
        ab_emb: torch.Tensor,
        ag_mask: torch.Tensor,
        ab_mask: torch.Tensor,
        ag_ids: torch.Tensor,
        ab_ids: torch.Tensor,
        indices: list[int],
    ):
        """Initialize AgAbPairDataset.

        Parameters
        ----------
        ag_emb : torch.Tensor
            Antigen embeddings
        ab_emb : torch.Tensor
            Antibody embeddings
        ag_mask : torch.Tensor
            Antigen sequence padding masks
        ab_mask : torch.Tensor
            Antibody sequence padding masks
        ag_ids: torch.Tensor
            Antigen global IDs
        ab_ids: torch.Tensor
            Antibody global IDs
        indices : list[int]
            Indices for the current split (train/val/test)
        """
        self.ag_emb = ag_emb
        self.ab_emb = ab_emb
        self.ag_mask = ag_mask
        self.ab_mask = ab_mask
        self.ag_ids = ag_ids
        self.ab_ids = ab_ids
        self.indices = torch.as_tensor(indices, dtype=torch.long)

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The number of samples in the dataset.
        """
        return int(self.indices.numel())

    def __getitem__(self, idx: int) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
    ]:
        """Get the data at index idx.

        Parameters
        ----------
        idx : int
            The index of the desired sample.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]
            A tuple of ``(ag_emb, ab_emb, ag_mask, ab_mask, ag_ids, ab_ids, global_index)``.
        """
        j = int(self.indices[idx])
        ag = self.ag_emb[j]
        ab = self.ab_emb[j]
        ag_mask = self.ag_mask[j]
        ab_mask = self.ab_mask[j]
        ag_ids = self.ag_ids[j]
        ab_ids = self.ab_ids[j]
        return ag, ab, ag_mask, ab_mask, ag_ids, ab_ids, j
