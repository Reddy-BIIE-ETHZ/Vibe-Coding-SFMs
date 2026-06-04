"""Utilities for training the model."""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as f
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    LRScheduler,
    SequentialLR,
)
from torch.utils.data import DataLoader

from calm.utils.checkpoint import CheckpointManager


def get_single_positive_labels(batch_size: int) -> torch.Tensor:
    """Generate labels for single-positive contrastive learning.

    Parameters
    ----------
    batch_size : int
        The size of the current batch.

    Returns
    -------
    torch.Tensor
        The tensor of labels for diagonal pairing: [0, 1, 2, ..., batch_size-1].
    """
    # Each sample is paired with itself (diagonal positive)
    labels = torch.arange(batch_size)
    return labels


@torch.no_grad()
def get_multi_positive_masks(
    ag_ids: torch.Tensor, ab_ids: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Generate multi-positive masks for contrastive learning.

    Parameters
    ----------
    ag_ids: torch.Tensor
        (N,) global IDs for antigens in this batch
    ab_ids: torch.Tensor
        (N,) global IDs for antibodies in this batch

    Returns
    -------
    dict[str, torch.Tensor]
        A dictionary containing:
        - 'ag': (N,N) bool tensor, pos_mask_ag[i,j] = True if Ag i and Ab j are positives
        - 'ab': (N,N) bool tensor, pos_mask_ab[j,i] = True if Ab j and Ag i are positives
    """
    # Ag->Ab positives: compare Ab IDs row-wise against the diagonal Ab_i ID
    # pos_mask_ag[i, j] = (ab_ids[j] == ab_ids[i])
    pos_mask_ag = ab_ids.unsqueeze(0) == ab_ids.unsqueeze(1)

    # Ab->Ag positives: compare Ag IDs row-wise against the diagonal Ag_j ID
    # pos_mask_ab[j, i] = (ag_ids[i] == ag_ids[j])
    pos_mask_ab = ag_ids.unsqueeze(0) == ag_ids.unsqueeze(1)

    label_size = ag_ids.size(0)
    diag = torch.eye(label_size, dtype=torch.bool, device=ag_ids.device)
    pos_mask_ag |= diag
    pos_mask_ab |= diag
    return {"ag": pos_mask_ag, "ab": pos_mask_ab}


@torch.no_grad()
def get_multi_positive_masks_nnk(
    ag_ids: torch.Tensor, ab_ids: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Generate multi-positive masks for contrastive learning.

    Parameters
    ----------
    ag_ids: torch.Tensor
        (N,) global IDs for antigens in this batch
    ab_ids: torch.Tensor
        (N,) global IDs for antibodies in this batch

    Returns
    -------
    dict[str, torch.Tensor]
        A dictionary containing:
        - 'ag': (N,N) bool tensor, pos_mask_ag[i,j] = True if Ag i and Ab j are positives
        - 'ab': (N,N) bool tensor, pos_mask_ab[j,i] = True if Ab j and Ag i are positives
    """
    # Ag->Ab positives: compare Ab IDs row-wise against the diagonal Ab_i ID
    # pos_mask_ag[i, j] = (ab_ids[j] == ab_ids[i]) or (ag_ids[i] == ag_ids[j])
    # (ab_ids[j] == ab_ids[i]): to label same Ab sequences
    # (ag_ids[i] == ag_ids[j]): to label same Ag sequences with different Ab sequences (NNK)
    pos_mask_ag = (ab_ids.unsqueeze(0) == ab_ids.unsqueeze(1)) | (
        ag_ids.unsqueeze(0) == ag_ids.unsqueeze(1)
    )

    # Ab->Ag positives: compare Ag IDs row-wise against the diagonal Ag_j ID
    # pos_mask_ab[j, i] = (ag_ids[i] == ag_ids[j])
    pos_mask_ab = (ag_ids.unsqueeze(0) == ag_ids.unsqueeze(1)) | (
        ab_ids.unsqueeze(0) == ab_ids.unsqueeze(1)
    )

    label_size = ag_ids.size(0)
    diag = torch.eye(label_size, dtype=torch.bool, device=ag_ids.device)
    pos_mask_ag |= diag
    pos_mask_ab |= diag
    return {"ag": pos_mask_ag, "ab": pos_mask_ab}


def compute_multi_positive_loss(
    logits_ag: torch.Tensor,  # (N,N) Ag->Ab
    logits_ab: torch.Tensor,  # (N,N) Ab->Ag
    pos_mask_ag: torch.Tensor,  # (N,N) bool
    pos_mask_ab: torch.Tensor,  # (N,N) bool
) -> torch.Tensor:
    """Compute the multi-positive contrastive loss.

    Parameters
    ----------
    logits_ag : torch.Tensor
        The ag->ab logits
    logits_ab : torch.Tensor
        The ab->ag logits
    pos_mask_ag : torch.Tensor
        The positive mask for ag->ab
    pos_mask_ab : torch.Tensor
        The positive mask for ab->ag

    Returns
    -------
    torch.Tensor
        The computed loss.
    """

    def one_side(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute the mean soft-label cross-entropy loss for one direction."""
        tgt = mask.float()
        counts = tgt.sum(dim=1, keepdim=True).clamp_min_(1.0)
        tgt = tgt / counts
        logp = f.log_softmax(logits, dim=1)
        loss = -(tgt * logp).sum(dim=1)
        return loss.mean()

    return 0.5 * (one_side(logits_ag, pos_mask_ag) + one_side(logits_ab, pos_mask_ab))


def compute_loss(logits: dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
    """Compute the symmetric cross-entropy contrastive loss.

    Parameters
    ----------
    logits : dict[str, torch.Tensor]
        The logits output from the model. Expected keys are:
            - 'ag': logits for antigen-to-antibody of shape (N, N).
            - 'ab': logits for antibody-to-antigen of shape (N, N).
    labels : torch.Tensor
        1D tensor of target class indices for diagonal cross-entropy loss.

    Returns
    -------
    torch.Tensor
        The averaged symmetric contrastive loss.
    """
    logits_per_ag = logits["ag"]
    logits_per_ab = logits["ab"]

    loss_fn = torch.nn.CrossEntropyLoss()
    loss_ag: torch.Tensor = loss_fn(logits_per_ag, labels)
    loss_ab: torch.Tensor = loss_fn(logits_per_ab, labels)
    return (loss_ag + loss_ab) / 2.0


def params_with_grad(m: torch.nn.Module) -> list[torch.nn.Parameter]:
    """Get model parameters that require gradients.

    Parameters
    ----------
    m : torch.nn.Module
        The model from which to extract parameters.

    Returns
    -------
    list[torch.nn.Parameter]
        A list of model parameters that require gradients.
    """
    return [p for p in m.parameters() if p.grad is not None]


def grads_all_finite(ps: list[torch.nn.Parameter]) -> bool:
    """Check if all gradients in the parameter list are finite.

    Parameters
    ----------
    ps : list[torch.nn.Parameter]
        List of model parameters to check for finite gradients.

    Returns
    -------
    bool
        True if all gradients are finite, False if any gradient contains
        inf or nan values.
    """
    return all(torch.isfinite(p.grad).all() for p in ps if p.grad is not None)


def build_optimizer(
    cfg: DictConfig, params: Iterable[torch.nn.Parameter]
) -> torch.optim.Optimizer:
    """Build the optimizer.

    Parameters
    ----------
    cfg : DictConfig
        The configuration object containing all settings for the pipeline.
    params : Iterable[torch.nn.Parameter]
        The model parameters to optimize.

    Returns
    -------
    torch.optim.Optimizer
        The constructed optimizer.
    """
    optimizer = cfg.optimizer
    lr = cfg.learning_rate
    weight_decay = cfg.weight_decay

    if optimizer == "AdamW":
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {optimizer}")


def build_scheduler(
    cfg: DictConfig,
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int | None = None,
) -> LRScheduler:
    """Build the scheduler.

    Parameters
    ----------
    cfg : DictConfig
        The configuration object containing all settings for the pipeline.
    optimizer : torch.optim.Optimizer
        The optimizer for which to build the scheduler.
    steps_per_epoch : int, optional
        Number of steps per epoch, required for certain schedulers.

    Returns
    -------
    LRScheduler
        The constructed learning rate scheduler.
    """
    scheduler = cfg.scheduler

    if scheduler == "CosineAnnealingWarmRestarts":
        return CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg.optimizer_T0, T_mult=cfg.optimizer_Tmult
        )
    elif scheduler == "LinearWarmupCosineDecay":
        # warmup / decay split
        num_training_steps = cfg.num_epochs * steps_per_epoch
        num_warmup_steps = max(1, int(cfg.warmup_ratio * num_training_steps))
        num_decay_steps = max(1, num_training_steps - num_warmup_steps)

        # 1) linear warmup from 0 -> base_lr
        warmup = LinearLR(
            optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=num_warmup_steps,
        )

        # 2) cosine anneal from base_lr -> ETA_MIN (NO restarts)
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=num_decay_steps,  # full remaining steps
            eta_min=1e-6,
        )

        return SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[num_warmup_steps]
        )
    raise ValueError(f"Unknown scheduler: {scheduler}")


def initialize(
    cfg: DictConfig, split_phase: bool = True, merge_train_val: bool = False
) -> tuple[
    dict[str, DataLoader[Any]],
    str,
    torch.nn.Module,
    torch.optim.Optimizer | None,
    LRScheduler | None,
    CheckpointManager | None,
    Any,
]:
    """Set up the training environment, including data loaders, model, optimizer, scheduler, and checkpoint manager.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing training parameters.
    split_phase : bool, optional
        Whether to split the dataset into training and validation sets, by default True.
    merge_train_val : bool, optional
        Whether to merge training and validation datasets, by default False.

    Returns
    -------
    tuple
        A tuple of seven elements:
            - dataloaders : dict[str, DataLoader]
                DataLoader objects for each phase.
            - out_dir : str
                Output directory for saving checkpoints and logs.
            - model : nn.Module
                The initialized model moved to the target device.
            - optimizer : Optimizer | None
                The optimizer for training (None during eval).
            - scheduler : LRScheduler | None
                The learning rate scheduler (None during eval).
            - checkpoint_manager : CheckpointManager | None
                Manager for saving and loading checkpoints (None during eval).
            - scaler : GradScaler | None
                Gradient scaler for mixed-precision training, or None if disabled.
    """
    stage = cfg.stage
    mode = cfg.mode

    device = cfg.gpu if torch.cuda.is_available() else "cpu"
    fp16 = cfg.fp16
    if fp16 and device != "cpu":
        from torch.cuda.amp import GradScaler

        scaler = GradScaler()
    else:
        scaler = None
        if fp16 and device == "cpu":
            print(
                "Warning: FP16 training on CPU is not recommended. Falling back to FP32."
            )

    # initialization based on stage
    if stage == "encoder":
        from calm.encoder.model import build_encoder_model

        out_dir = cfg.paths.encoder_dir_fold
        checkpoint_cfg = cfg.train.encoder

        # Use the index-based loader when configured (tSFM/eSFM/dtSFM), else the
        # default pair-based loader (crisprSFM/mir-SFM/mhcSFM).
        loader_type = getattr(cfg.data.db, "data_loader_type", "default")
        if loader_type == "tsfm":
            from calm.encoder.data_tsfm import build_tsfm_dataloaders
            dataloaders = build_tsfm_dataloaders(
                cfg, split_phase=split_phase, merge_train_val=merge_train_val
            )
        else:
            from calm.encoder.data import build_encoder_dataloaders
            dataloaders = build_encoder_dataloaders(
                cfg, split_phase=split_phase, merge_train_val=merge_train_val
            )
        model = build_encoder_model(cfg).to(device)

        # Load pretrained checkpoint if specified (for fine-tuning)
        pretrained_ckpt = getattr(checkpoint_cfg, "pretrained_checkpoint", None)
        if pretrained_ckpt:
            print(f"Loading pretrained checkpoint: {pretrained_ckpt}")
            load_device = f"cuda:{device}" if isinstance(device, int) else device
            state = torch.load(pretrained_ckpt, map_location=load_device, weights_only=True)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print(f"  Missing keys ({len(missing)}): {missing[:5]}...")
            if unexpected:
                print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
            print(f"  Pretrained checkpoint loaded successfully")

    elif stage == "decoder":
        out_dir = cfg.paths.decoder_dir_fold
        checkpoint_cfg = cfg.train.decoder
        from calm.decoder.data import build_decoder_dataloaders, load_tokenizer
        from calm.decoder.model import build_decoder_model
        dataloaders = build_decoder_dataloaders(
            cfg, split_phase=split_phase, merge_train_val=merge_train_val
        )
        tokenizer = load_tokenizer(cfg)
        model = build_decoder_model(cfg, tokenizer).to(device)

    # training setup
    os.makedirs(out_dir, exist_ok=True)

    if mode == "train":
        optimizer = build_optimizer(checkpoint_cfg, model.parameters())
        scheduler = build_scheduler(
            checkpoint_cfg, optimizer, steps_per_epoch=len(dataloaders["train"])
        )
        checkpoint_manager = CheckpointManager(checkpoint_cfg, out_dir=out_dir)
    elif mode == "eval":
        optimizer = None
        scheduler = None
        checkpoint_manager = None

    return (
        dataloaders,
        out_dir,
        model,
        optimizer,
        scheduler,
        checkpoint_manager,
        scaler,
    )


def setup_logging(
    cfg: DictConfig, model: torch.nn.Module, project_name: str, run_name: str
) -> None:
    """Set up logging for the training process using Weights & Biases (wandb).

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing training parameters.
    model : torch.nn.Module
        The model to be logged and monitored.
    project_name : str
        The name of the wandb project.
    run_name : str
        The name of the specific wandb run.
    """
    import wandb
    import yaml
    from omegaconf import OmegaConf

    wandb.init(project=project_name)
    cfg_clean = yaml.safe_load(OmegaConf.to_yaml(cfg, resolve=True))
    wandb.config.update(cfg_clean)  # type: ignore[no-untyped-call]
    if wandb.run is not None:
        wandb.run.name = run_name

    wandb.watch(model)
