"""Evaluation loop for CALM Stage-1."""

import os

import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

from calm.utils.train_utils import get_multi_positive_masks_nnk, initialize


def get_best_model_checkpoint(dir_eval: str) -> str:
    """Find the best model checkpoint from the corresponding training directory.

    Given an evaluation output directory, locates the training directory and
    returns the path to the best model checkpoint based on validation accuracy.

    Parameters
    ----------
    dir_eval : str
        Evaluation output directory path.

    Returns
    -------
    str
        Full path to the best model checkpoint file.
    """
    # Convert eval directory path to corresponding train directory path
    parent_dir = os.path.dirname(dir_eval)

    if os.path.basename(parent_dir) != "eval":
        raise ValueError("Expected parent directory to be named 'eval'.")

    grandparent_dir = os.path.dirname(parent_dir)
    fold_name = os.path.basename(dir_eval)
    dir_train = os.path.join(grandparent_dir, "train", fold_name)

    if not os.path.exists(dir_train):
        raise FileNotFoundError(f"Training directory not found: {dir_train}")

    # Collect the final model checkpoint.
    # Priority 1: final_model_*.pth (written at end of retrain phase).
    # Priority 2: highest-epoch best_model_val_pred_acc_*.pth fallback —
    #   used when the retrain phase fails to write a final model (see
    #   audit/tsfm/SCOPING.md and the retrain-silent-exit note in
    #   audit/PRESERVATION_DISCIPLINE.md). The best-val checkpoint is
    #   the same weights the retrain would start from, so eval is valid.
    checkpoint_name = [f for f in os.listdir(dir_train) if f.startswith("final_model")]
    if checkpoint_name:
        if len(checkpoint_name) > 1:
            raise ValueError(f"Expected exactly one match, found {len(checkpoint_name)}")
        chosen = checkpoint_name[0]
    else:
        best_val = sorted(
            (f for f in os.listdir(dir_train) if f.startswith("best_model_val_pred_acc_epoch_")),
            key=lambda n: int(n.replace("best_model_val_pred_acc_epoch_", "").replace(".pth", "")),
        )
        if not best_val:
            raise FileNotFoundError(
                f"No final_model_*.pth or best_model_val_pred_acc_*.pth found in {dir_train}"
            )
        chosen = best_val[-1]
        print(
            f"  WARN: no final_model_*.pth in {dir_train}; falling back to "
            f"best-val checkpoint: {chosen}"
        )

    checkpoint_path = os.path.join(dir_train, chosen)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    return checkpoint_path


def run_eval(
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    model: nn.Module,
    out_dir: str,
) -> None:
    """Run evaluation and collect model outputs.

    Parameters
    ----------
    dataloader : DataLoader
        The dataloader for the current phase.
    model : nn.Module
        The model to evaluate.
    out_dir : str
        Directory to save evaluation outputs.
    """
    # Initialize collections for predictions and outputs
    all_features_ag: torch.Tensor
    all_features_ab: torch.Tensor
    all_projections_ag: list[torch.Tensor] = []
    all_projections_ab: list[torch.Tensor] = []
    all_ids_ag: list[torch.Tensor] = []
    all_ids_ab: list[torch.Tensor] = []
    feat_ag_list: list[torch.Tensor] = []
    feat_ab_list: list[torch.Tensor] = []

    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in dataloader:
            # Defensive batch unpacking (mhcSFM v2 Phase 1.5b): dataset
            # returns 7-tuple (v1) or 8-tuple (v2 with per-source temperature).
            ag_emb, ab_emb, ag_mask, ab_mask, ag_ids, ab_ids, indices = batch[:7]
            source_idx = batch[7] if len(batch) > 7 else None

            ag_emb = ag_emb.to(device)
            ab_emb = ab_emb.to(device)
            ag_mask = ag_mask.to(device)
            ab_mask = ab_mask.to(device)
            ag_ids = ag_ids.to(device)
            ab_ids = ab_ids.to(device)
            if source_idx is not None:
                source_idx = source_idx.to(device)

            _, _, _, outputs = model(
                ag_emb, ab_emb, ag_mask, ab_mask, source_idx=source_idx
            )

            # Collect model outputs
            feat_ag_list.append(outputs["features_ag"].cpu())
            feat_ab_list.append(outputs["features_ab"].cpu())
            all_projections_ag.append(outputs["projections_ag"].cpu())
            all_projections_ab.append(outputs["projections_ab"].cpu())
            all_ids_ag.append(ag_ids.cpu())
            all_ids_ab.append(ab_ids.cpu())

    # Concatenate all projections and move to GPU
    all_features_ag = torch.cat(feat_ag_list, dim=0).to(device)
    all_features_ab = torch.cat(feat_ab_list, dim=0).to(device)
    cosine_sim = all_features_ag @ all_features_ab.t()

    all_features_ag = all_features_ag.cpu()
    all_features_ab = all_features_ab.cpu()
    cosine_sim = cosine_sim.cpu()

    ids_ag = torch.cat(all_ids_ag, dim=0)
    ids_ab = torch.cat(all_ids_ab, dim=0)
    labels_masks = get_multi_positive_masks_nnk(ids_ag, ids_ab)

    # Save model outputs
    features = {
        "ag": all_features_ag,
        "ab": all_features_ab,
    }
    file_features = os.path.join(out_dir, "features.pt")
    torch.save(features, file_features)

    projections = {
        "ag": torch.cat(all_projections_ag, dim=0),
        "ab": torch.cat(all_projections_ab, dim=0),
    }
    file_projections = os.path.join(out_dir, "projections.pt")
    torch.save(projections, file_projections)

    file_cosine_sim = os.path.join(out_dir, "cosine_sim.pt")
    torch.save(cosine_sim, file_cosine_sim)

    file_labels_masks = os.path.join(out_dir, "labels_masks.pt")
    torch.save(labels_masks, file_labels_masks)


def evaluate_fold(cfg: DictConfig) -> None:
    """Evaluate model and collect outputs for specified phases.

    Parameters
    ----------
    cfg : DictConfig
        Configuration dictionary containing evaluation parameters.
    """
    phaselist = cfg.phaselist

    # initial setup
    (
        dataloaders,
        out_dir,
        model,
        _,
        _,
        _,
        _,
    ) = initialize(cfg, split_phase=False)

    # Load model
    device = next(model.parameters()).device
    path_model_weight = cfg.eval.encoder.model_weight
    if path_model_weight is None or os.path.exists(path_model_weight) is False:
        path_model_weight = get_best_model_checkpoint(out_dir)

    state_dict = torch.load(path_model_weight, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loading model weights from: {path_model_weight}")

    # Run evaluation
    if not isinstance(dataloaders, dict):
        raise TypeError("dataloaders must be a dict in eval mode; check configuration.")
    for phase in phaselist:
        run_eval(dataloaders[phase], model, out_dir)

    print("Evaluation completed successfully.")
