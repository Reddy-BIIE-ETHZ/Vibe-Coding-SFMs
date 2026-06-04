"""Training loop for CALM Stage-1."""

from __future__ import annotations

import os

import torch
from omegaconf import DictConfig
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from calm.encoder.metrics import MetricsTracker, save_results
from calm.utils.train_utils import (
    compute_multi_positive_loss,
    get_multi_positive_masks_nnk,
    initialize,
    setup_logging,
)


def train(cfg: DictConfig) -> None:
    """Train the model based on the provided configuration.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing all settings for training.
    """
    deploy = cfg.deploy
    if deploy:
        train_deploy(cfg)
    else:
        train_fold(cfg)


def train_fold(cfg: DictConfig) -> None:
    """Train a single fold of the model.

    Parameters
    ----------
    cfg : DictConfig
        Configuration dictionary containing training parameters.
    """
    # 1. Train with validation to get best epoch
    run_name = cfg.run_name + f"-fold-{cfg.fold}-dev"

    best_epoch = run_training_loop(cfg=cfg, run_name=run_name, return_best_epoch=True)
    if best_epoch is None:
        raise RuntimeError("run_training_loop did not return a best epoch")
    print(f"Best epoch from validation set: {best_epoch}")

    # 2. Train with train+val using best epoch
    cfg.phaselist = ["train", "test"]
    cfg.train.encoder.checkpoint_phase = "test"
    cfg.train.encoder.num_epochs = best_epoch + 1
    print("Forced configurations:")
    print(f"  - phaselist: {cfg.phaselist}")
    print(f"  - checkpoint_phase: {cfg.train.encoder.checkpoint_phase}")
    print(f"  - num_epochs: {cfg.train.encoder.num_epochs}")

    run_name = cfg.run_name + f"-fold-{cfg.fold}-test"
    merge_train_val = True
    run_training_loop(
        cfg=cfg,
        run_name=run_name,
        merge_train_val=merge_train_val,
        save_final_model=True,
    )


def train_deploy(cfg: DictConfig) -> None:
    """Train the model on the full training data for deployment.

    Parameters
    ----------
    cfg : DictConfig
        Configuration dictionary containing training parameters.
    """
    run_name = cfg.run_name + "-deploy"
    run_training_loop(
        cfg=cfg,
        run_name=run_name,
        split_phase=False,
        save_final_model=True,
    )


def run_epoch(
    phase: str,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    model: nn.Module,
    optimizer: Optimizer | None,
    scheduler: LRScheduler | None,
    max_norm_grad: float = 1.0,
) -> dict[str, float]:
    """Run a single epoch of training or evaluation.

    Parameters
    ----------
    phase : str
        The current phase one of either ["train", "val", "test"].
    dataloader : DataLoader
        The dataloader for the current phase.
    model : nn.Module
        The model to train or evaluate.
    optimizer : Optimizer, optional
        The optimizer for training, by default None
    scheduler : _type_, optional
        The learning rate scheduler for training, by default None
    grad_accumulation_steps : int, optional
        Number of gradient accumulation steps, by default 1
    max_norm_grad : float, optional
        Maximum gradient norm for clipping, by default 1.0

    Returns
    -------
    dict
        A dictionary containing epoch results with flexible metrics
    """
    # Initialize metrics tracker
    metrics = MetricsTracker()

    # Set model mode
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    # Main training/evaluation loop
    for batch in dataloader:
        if is_train:
            if optimizer is None:
                raise ValueError("optimizer must not be None during training")
            optimizer.zero_grad(set_to_none=True)

        # Defensive batch unpacking: dataset returns 7-tuple (v1 SFMs) or
        # 8-tuple (mhcSFM v2 with per-source temperature). The 8th element,
        # if present, is per-pair source_idx for the per-source temperature
        # path in CALMEncoder.forward (model.py Phase 1.3).
        ag_emb, ab_emb, ag_mask, ab_mask, ag_ids, ab_ids, indices = batch[:7]
        source_idx = batch[7] if len(batch) > 7 else None

        batch_size = ag_emb.shape[0]
        device = next(model.parameters()).device
        ag_emb = ag_emb.to(device)
        ab_emb = ab_emb.to(device)
        ag_mask = ag_mask.to(device)
        ab_mask = ab_mask.to(device)
        if source_idx is not None:
            source_idx = source_idx.to(device)

        ag_ids = ag_ids.to(device)
        ab_ids = ab_ids.to(device)
        labels = get_multi_positive_masks_nnk(ag_ids, ab_ids)
        labels = {k: v.to(device) for k, v in labels.items()}

        with torch.set_grad_enabled(is_train):
            # Forward pass — source_idx=None falls through to the v1 scalar
            # path in model.py and produces byte-identical math.
            logits, cosine_sim, logit_scale, _ = model(
                ag_emb, ab_emb, ag_mask, ab_mask, source_idx=source_idx
            )

            # Loss calculation
            loss = compute_multi_positive_loss(
                logits["ag"], logits["ab"], labels["ag"], labels["ab"]
            )

            grad_norm_value = 0.0  # default for eval
            # Backward pass (training only)
            if is_train and optimizer is not None and scheduler is not None:
                loss.backward()  # type: ignore[no-untyped-call]
                if max_norm_grad is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=max_norm_grad
                    )
                    grad_norm_value = float(
                        torch.nan_to_num(grad_norm, nan=0.0, posinf=1e9)
                    )
                optimizer.step()
                scheduler.step()

        # Update metrics for this batch
        metrics.update_batch(
            batch_size,
            loss.item(),
            logits,
            cosine_sim,
            labels,
            grad_norm_value,
        )

        # Add custom metrics
        metrics.add_custom_metric("logit_scale", logit_scale.item())
        if optimizer is not None:
            metrics.add_custom_metric("learning_rate", optimizer.param_groups[0]["lr"])

    # Get final epoch results
    results: dict[str, float] = metrics.get_epoch_results()

    return results


def run_epoch_grad_acc(
    phase: str,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    model: nn.Module,
    optimizer: Optimizer | None,
    scheduler: LRScheduler | None,
    max_norm_grad: float = 1.0,
    grad_accumulation_steps: int = 1,
    accumulation_counter: int = 0,
    current_step: int = 0,
    num_epochs: int = 0,
) -> tuple[dict[str, float], int, int]:
    """Run a single epoch of training or evaluation with gradient accumulation.

    Parameters
    ----------
    phase : str
        The current phase one of either ["train", "val", "test"].
    dataloader : DataLoader
        The dataloader for the current phase.
    model : nn.Module
        The model to train or evaluate.
    optimizer : Optimizer, optional
        The optimizer for training, by default None
    scheduler : LRScheduler, optional
        The learning rate scheduler for training, by default None
    max_norm_grad : float, optional
        Maximum gradient norm for clipping, by default 1.0
    grad_accumulation_steps : int, optional
        Number of epochs to accumulate gradients before updating, by default 1
    accumulation_counter : int, optional
        Counter for gradient accumulation across epochs, by default 0
    current_step : int, optional
        Current training step (for tracking across epochs), by default 0
    num_epochs : int, optional
        Total number of epochs (to handle last epoch), by default 0

    Returns
    -------
    tuple[dict[str, float], int, int]
        A tuple containing:
        - dict: A dictionary containing epoch results with flexible metrics
        - int: Updated current step count
        - int: Updated accumulation counter
    """
    # Initialize metrics tracker
    metrics = MetricsTracker()

    # Set model mode
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    # Main training/evaluation loop (single batch per epoch)
    for batch in dataloader:
        # Defensive batch unpacking (mhcSFM v2 Phase 1.5b): see comment at
        # line ~136 — same pattern, dataset returns 7- or 8-tuple.
        ag_emb, ab_emb, ag_mask, ab_mask, ag_ids, ab_ids, indices = batch[:7]
        source_idx = batch[7] if len(batch) > 7 else None

        batch_size = ag_emb.shape[0]
        device = next(model.parameters()).device
        ag_emb = ag_emb.to(device)
        ab_emb = ab_emb.to(device)
        ag_mask = ag_mask.to(device)
        ab_mask = ab_mask.to(device)
        if source_idx is not None:
            source_idx = source_idx.to(device)

        ag_ids = ag_ids.to(device)
        ab_ids = ab_ids.to(device)
        labels = get_multi_positive_masks_nnk(ag_ids, ab_ids)
        labels = {k: v.to(device) for k, v in labels.items()}

        with torch.set_grad_enabled(is_train):
            # Forward pass — source_idx=None falls through to v1 scalar path.
            logits, cosine_sim, logit_scale, _ = model(
                ag_emb, ab_emb, ag_mask, ab_mask, source_idx=source_idx
            )

            # Loss calculation
            loss = compute_multi_positive_loss(
                logits["ag"], logits["ab"], labels["ag"], labels["ab"]
            )

            # Scale loss for gradient accumulation
            if is_train and grad_accumulation_steps > 1:
                loss = loss / grad_accumulation_steps

            grad_norm_value = 0.0  # default for eval

            # Backward pass (training only)
            if is_train and optimizer is not None:
                loss.backward()  # type: ignore[no-untyped-call]

                # Increment accumulation counter
                accumulation_counter += 1

                # Only update weights after accumulating enough gradients across epochs
                # or on the last epoch
                should_update = (
                    accumulation_counter % grad_accumulation_steps == 0
                ) or (num_epochs > 0 and accumulation_counter == num_epochs)

                if should_update:
                    if max_norm_grad is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=max_norm_grad
                        )
                        grad_norm_value = float(
                            torch.nan_to_num(grad_norm, nan=0.0, posinf=1e9)
                        )

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    if scheduler is not None:
                        scheduler.step()

                    current_step += 1

        # Update metrics for this batch
        # Note: loss is scaled, so we need to unscale it for metrics
        actual_loss = (
            loss.item() * grad_accumulation_steps
            if (is_train and grad_accumulation_steps > 1)
            else loss.item()
        )

        metrics.update_batch(
            batch_size,
            actual_loss,
            logits,
            cosine_sim,
            labels,
            grad_norm_value,
        )

        # Add custom metrics
        metrics.add_custom_metric("logit_scale", logit_scale.item())
        if optimizer is not None:
            metrics.add_custom_metric("learning_rate", optimizer.param_groups[0]["lr"])

    # Get final epoch results
    results = metrics.get_epoch_results()

    return results, current_step, accumulation_counter


def run_training_loop(
    cfg: DictConfig,
    run_name: str,
    split_phase: bool = True,
    merge_train_val: bool = False,
    return_best_epoch: bool = False,
    save_final_model: bool = False,
) -> None | int:
    """Train a single fold of the model.

    Parameters
    ----------
    cfg : DictConfig
        The configuration object containing all training settings.
    run_name : str
        Name for the current training run (used for logging).
    split_phase : bool, optional
        Whether to split the dataset into train/val/test phases, by default True.
    merge_train_val : bool, optional
        Whether to merge training and validation datasets, by default False.
    return_best_epoch : bool, optional
        Whether to return the best epoch number after training, by default False.
    save_final_model : bool, optional
        Whether to save the model weights at the final epoch, by default False.

    Returns
    -------
    None | int
        Returns best epoch number if return_best_epoch is True, otherwise None
    """
    fold = cfg.fold
    project_name = cfg.project_name
    phaselist = cfg.phaselist
    train_cfg = cfg.train.encoder
    num_epochs = train_cfg.num_epochs
    use_grad_accumulation = train_cfg.use_grad_accumulation
    grad_accumulation_steps = train_cfg.grad_accumulation_steps

    # initial setup
    (
        dataloaders,
        out_dir,
        model,
        optimizer,
        scheduler,
        checkpoint_manager,
        _,
    ) = initialize(cfg, split_phase=split_phase, merge_train_val=merge_train_val)

    # logging setup
    if cfg.wandb:
        import wandb

        setup_logging(cfg, model, project_name, run_name)

    if checkpoint_manager is None:
        raise RuntimeError(
            "checkpoint_manager is None in train mode; check configuration."
        )

    if not isinstance(dataloaders, dict):
        raise TypeError(
            "dataloaders must be a dict in train mode; check configuration."
        )

    # initialize epoch storage for results
    best_epoch = 0
    epoch_results: dict[str, list[dict[str, float]]] = {
        phase: [] for phase in phaselist
    }
    if use_grad_accumulation:
        current_step = 0  # Track training steps across epochs
        accumulation_counter = 0  # Track accumulation across epochs
        previous_step = -1  # Track previous step to detect changes

    for epoch in range(num_epochs):
        # Store results for all phases in this epoch
        epoch_phase_results = {}

        for phase in phaselist:
            # run epoch and get results
            if use_grad_accumulation:
                results, current_step, accumulation_counter = run_epoch_grad_acc(
                    phase,
                    dataloaders[phase],
                    model,
                    optimizer,
                    scheduler,
                    train_cfg.max_norm_grad,
                    grad_accumulation_steps,
                    accumulation_counter,
                    current_step,
                    num_epochs,
                )
            else:
                results = run_epoch(
                    phase,
                    dataloaders[phase],
                    model,
                    optimizer,
                    scheduler,
                    train_cfg.max_norm_grad,
                )

            # log epoch results
            results["epoch"] = epoch
            results["phase"] = phase
            if use_grad_accumulation:
                results["steps"] = current_step

            # Store results for later logging
            epoch_phase_results[phase] = results

            # save results
            epoch_results[phase].append(results)

            # log summary for this phase
            print(
                f"Epoch {epoch} {phase}: Loss={results['loss']:.4f}, "
                f"Logits_avg={results['avg_logits']:.4f}, "
                f"Acc_avg={results['pred_acc']:.4f}"
            )

            # mhcSFM v2 Phase 1.6 diagnostic: log per-source temperatures
            # whenever the model has more than one source. Helps detect tau
            # explosion / collapse during training (per-source path in
            # CALMEncoder, model.py Phase 1.3 design).
            if (
                phase == "train"
                and hasattr(model, "n_sources")
                and model.n_sources > 1
            ):
                logit_scale_vals = model.logit_scale.detach().cpu().tolist()
                tau_vals = [
                    1.0 / min(model.max_scale, float(torch.exp(model.logit_scale[i]).item()))
                    for i in range(model.n_sources)
                ]
                print(
                    f"  per-source logit_scale: "
                    f"{[f'{v:.4f}' for v in logit_scale_vals]}  "
                    f"=> tau: {[f'{t:.4f}' for t in tau_vals]}"
                )

        # Log to wandb after all phases complete (only if step increased)
        if cfg.wandb:
            if use_grad_accumulation:
                # Only log when step has increased
                if current_step > previous_step:
                    for phase, results in epoch_phase_results.items():
                        log_dict = {f"{phase}/{k}": v for k, v in results.items()}
                        wandb.log(log_dict, step=current_step)
                    previous_step = current_step
            else:
                for phase, results in epoch_phase_results.items():
                    log_dict = {f"{phase}/{k}": v for k, v in results.items()}
                    wandb.log(log_dict, step=epoch)

        # checkpoint if best metric is achieved
        is_better = checkpoint_manager.check_and_save(epoch_results, model, epoch)
        if is_better:
            best_epoch = epoch
            print(f"Checkpoint saved at fold {fold} - epoch {epoch}")

        # save final model
        if save_final_model and (epoch == num_epochs - 1):
            save_path = os.path.join(out_dir, f"final_model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), save_path)

    if cfg.wandb:
        wandb.finish()

    # print best checkpoint summary
    checkpoint_manager.summary()
    save_results(
        epoch_results,
        file_out=os.path.join(out_dir, f"results_{'_'.join(cfg.phaselist)}.csv"),
    )

    if return_best_epoch:
        return best_epoch
    return None
