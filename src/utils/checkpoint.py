"""
Checkpoint manager for model training.

Tracks specified metric and saves model when it improves.
"""

from __future__ import annotations

import os

import torch
from omegaconf import DictConfig


class CheckpointManager:
    """Manages model checkpoints.

    Tracks a specified metric from a specified phase and saves the model
    when the metric improves.
    """

    def __init__(self, cfg: DictConfig, out_dir: str):
        """Initialize checkpoint manager.

        Parameters
        ----------
        cfg : DictConfig
            Arguments containing checkpoint configuration.
        out_dir : str
            Directory to save checkpoints.
        """
        self.out_dir = out_dir
        self.save_path_best = os.path.join(self.out_dir, "best_model.pth")

        self.phase = cfg.checkpoint_phase
        self.metric = cfg.checkpoint_metric
        self.mode = cfg.checkpoint_mode
        self.save_every = cfg.checkpoint_save_every

        # Initialize best metric
        self.best_metric = float("-inf") if self.mode == "max" else float("inf")
        self.best_epoch = -1

    def check_and_save(
        self,
        epoch_results: dict[str, list[dict[str, float]]],
        model: torch.nn.Module,
        epoch: int,
    ) -> bool:
        """Check if current results are better and save model weights if so.

        Parameters
        ----------
        epoch_results : dict
            Dictionary with results for each phase
                - "train": list of dicts with metrics for each training epoch
                - "val": list of dicts with metrics for each validation epoch
                - "test": list of dicts with metrics for each test epoch
        model : torch.nn.Module
            Model to save
        epoch : int, optional
            Current epoch number for logging

        Returns
        -------
        bool
            True if model was saved, False otherwise
        """
        # Check if we have results for the monitored phase
        if self.phase not in epoch_results or len(epoch_results[self.phase]) == 0:
            raise ValueError(f"Missing results for phase '{self.phase}'")

        # Get latest results for the phase
        current_results = epoch_results[self.phase][-1]

        # Check if metric exists in results
        if self.metric not in current_results:
            raise ValueError(
                f"Missing results for metric '{self.metric}' in phase '{self.phase}'"
            )

        current_metric = current_results[self.metric]

        # Save regular checkpoint every N epochs
        if epoch % self.save_every == 0:
            regular_path = os.path.join(
                self.out_dir, f"model_checkpoint_epoch_{epoch}.pth"
            )
            torch.save(model.state_dict(), regular_path)

        # Determine if this is better
        is_better = self._is_better(current_metric)

        if is_better:
            self.best_metric = current_metric
            self.best_epoch = epoch

            # Save best model
            self.save_path = os.path.join(
                self.out_dir, f"best_model_{self.phase}_{self.metric}_epoch_{epoch}.pth"
            )
            torch.save(model.state_dict(), self.save_path)

            return True

        return False

    def _is_better(self, current_metric: float) -> bool:
        """Check if current metric is better than best.

        Parameters
        ----------
        current_metric : float
            Current metric value to compare

        Returns
        -------
        bool
            True if current metric is better than best, False otherwise
        """
        if self.mode == "max":
            return current_metric > self.best_metric
        else:
            return current_metric < self.best_metric

    def summary(self) -> None:
        """Get information about the best checkpoint."""
        print(
            f"Best {self.metric} achieved at epoch {self.best_epoch}: {self.best_metric:.4f}"
        )
        print(f"Best model saved to: {self.save_path_best}")
