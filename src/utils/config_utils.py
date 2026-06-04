"""Utility functions for configuration management in the CALM pipeline."""

from __future__ import annotations

import copy
import os
import random
import sys

import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


def setup_config(cfg: DictConfig) -> None:
    """Set up the configuration.

    Parameters
    ----------
    cfg : DictConfig
        The configuration object containing all settings for the pipeline.
    """
    stage = cfg.stage
    mode = cfg.mode

    try:
        hydra_cfg = HydraConfig.get()
        output_dir = hydra_cfg.runtime.output_dir
    except Exception:
        # Fallback if Hydra is not initialized
        sys.exit("Could not access Hydra output directory.")

    if stage == "encoder":
        cfg.paths.encoder_dir = output_dir
    elif stage == "decoder":
        cfg.paths.decoder_dir = output_dir
        if not os.path.exists(cfg.paths.encoder_dir):
            raise FileNotFoundError(
                f"The encoder directory {cfg.paths.encoder_dir} does not exist."
            )

    if mode == "eval":
        # Set evaluation-specific configurations.
        cfg.shuffle_train = False
        cfg.wandb = False
        cfg.phaselist = ["test"]
        cfg.train.encoder.reduce_embeddings = False

        print("[ANALYSIS MODE] Forced configurations:")
        print(f"  - shuffle_train: {cfg.shuffle_train}")
        print(f"  - wandb: {cfg.wandb}")
        print(f"  - phaselist: {cfg.phaselist}")
        print(
            f"  - train.encoder.reduce_embeddings: {cfg.train.encoder.reduce_embeddings}"
        )

    if cfg.deploy:
        cfg.phaselist = ["train"]
        cfg.fold_list = [0]
        cfg.train.encoder.checkpoint_phase = "train"
        cfg.train.decoder.checkpoint_phase = "train"

        print("Forced configurations:")
        print(f"  - phaselist: {cfg.phaselist}")
        print(f"  - fold_list: {cfg.fold_list}")


def validate_config(cfg: DictConfig) -> None:
    """Validate the configuration parameters.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing all settings for the pipeline.
    """
    try:
        stage = cfg.stage
        mode = cfg.mode
    except AttributeError:
        sys.exit(
            "Both 'stage' and 'mode' must be specified in the CLI. "
            "Use: python main.py stage=<encoder|decoder|all> mode=<train|eval>"
        )

    valid_stages = ["encoder", "decoder"]
    valid_modes = ["train", "eval"]

    if stage not in valid_stages:
        raise ValueError(f"Invalid stage '{stage}'. Must be one of: {valid_stages}")

    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of: {valid_modes}")

    # Validate model dim
    if cfg.model.encoder.d_model != cfg.model.decoder.d_model:
        raise ValueError(
            f"Encoder and Decoder d_model must match. "
            f"Got encoder d_model={cfg.model.encoder.d_model} and decoder d_model={cfg.model.decoder.d_model}"
        )

    # Validate checkpoint phase
    if mode == "train":
        phaselist = cfg.phaselist  # Use the global phaselist

        if stage == "encoder":
            checkpoint_phase = cfg.train.encoder.checkpoint_phase
        elif stage == "decoder":
            checkpoint_phase = cfg.train.decoder.checkpoint_phase

        if checkpoint_phase not in phaselist:
            raise ValueError(
                f"Checkpoint_phase must be one of {phaselist}, got {checkpoint_phase}"
            )


def setup_environment(cfg: DictConfig) -> None:
    """Set up the environment based on configuration.

    Parameters
    ----------
    cfg : DictConfig
        The configuration object containing all settings for the pipeline.
    """
    # Set up GPU/CPU device
    if cfg.gpu != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)

    # Set random seed for reproducibility
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.random_seed)
        torch.cuda.manual_seed_all(cfg.random_seed)


def update_config(
    cfg: DictConfig,
    fold: int,
) -> DictConfig:
    """Update cfg for the current fold.

    Parameters
    ----------
    cfg : DictConfig
        The original arguments.
    fold : int
        The current fold number.

    Returns
    -------
    cfg_fold : DictConfig
        The updated arguments for the current fold.
    """
    cfg_fold = copy.deepcopy(cfg)
    cfg_fold.fold = fold

    cfg_fold.paths.encoder_dir_fold = os.path.join(
        cfg.paths.encoder_dir, f"fold_{fold}"
    )

    if cfg.paths.decoder_dir is not None:
        cfg_fold.paths.decoder_dir_fold = os.path.join(
            cfg.paths.decoder_dir, f"fold_{fold}"
        )
    else:
        cfg_fold.paths.decoder_dir_fold = None

    cfg_fold.paths.file_index_fold = os.path.join(
        cfg_fold.paths.index_dir,
        f"split_hash_ids_outerfold_{fold}_innerfold_{fold}.json",
    )

    return cfg_fold
