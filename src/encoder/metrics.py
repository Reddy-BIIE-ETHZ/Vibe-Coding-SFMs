"""
MetricsTracker class for flexible epoch metrics tracking.

Handles standard metrics (incl. loss, accuracy) and custom metrics with automatic averaging.
"""

from __future__ import annotations

import csv
import math

import torch


class MetricsTracker:
    """
    Flexible metrics tracker for training/evaluation epochs.

    Handles both standard metrics and custom metrics with automatic averaging.
    """

    def __init__(self, ef_alpha: float = 0.01) -> None:
        """Initialize the MetricsTracker.

        Parameters
        ----------
        ef_alpha : float, optional
            Fraction used for the Enrichment Factor (EF) calculation, by default 0.01.
        """
        # k values for recall/precision@k
        self.k_values = [1, 5, 10, 30, 50]

        # Enrichment Factor (EF) alpha
        self.ef_alpha = float(ef_alpha)

        self.reset()

    def reset(self) -> None:
        """Reset all metrics for a new epoch."""
        self.total_loss = 0.0
        self.total_samples = 0
        self.batch_count = 0

        # Accuracy tracking
        self.correct_ag = 0
        self.correct_ab = 0

        # Recall/Precision@k tracking (dict: k -> {'ag': float, 'ab': float})
        self.hits_at_k = {k: {"ag": 0.0, "ab": 0.0} for k in self.k_values}
        self.items_at_k: dict[int, dict[str, list[torch.Tensor]]] = {
            k: {"ag": [], "ab": []} for k in self.k_values
        }

        # Micro-averaged EF components (sum over queries)
        ## direction: ag->ab
        self._ef_num_ag2ab = 0.0  # sum_i a_i
        self._ef_den_ag2ab = 0.0  # sum_i A_i

        ## direction: ab->ag
        self._ef_num_ab2ag = 0.0
        self._ef_den_ab2ag = 0.0

        # Logits & cosine similarity tracking
        self.total_logits_ag = 0.0
        self.total_logits_ab = 0.0

        self.total_cosine_sim = 0.0
        self.total_margin_max_neg_ag = 0.0
        self.total_margin_max_neg_ab = 0.0

        self.total_grad_norm = 0.0
        self.max_grad_norm = 0.0

        # MRR tracking
        self.sum_rr_ag2ab = 0.0
        self.sum_rr_ab2ag = 0.0
        self.count_rr_ag2ab = 0
        self.count_rr_ab2ag = 0
        self.sum_rank_ag2ab = 0.0
        self.sum_rank_ab2ag = 0.0
        self.max_rank_ag2ab = 0
        self.max_rank_ab2ag = 0

        # Additional metrics storage
        self.custom_metrics: dict[str, list[float]] = {}

    def update_batch(
        self,
        batch_size: int,
        loss_val: float,
        logits: dict[str, torch.Tensor],
        cosine_sim: torch.Tensor,
        labels: dict[str, torch.Tensor],
        grad_norm: float,
    ) -> None:
        """Update metrics for current batch.

        Parameters
        ----------
        batch_size : int
            Size of the current batch.
        loss_val : float
            Loss value for this batch.
        logits : dict[str, torch.Tensor]
            Dictionary containing logits tensors:
                - "ag": logits from antigen-to-antibody of shape (N, N).
                - "ab": logits from antibody-to-antigen of shape (N, N).
        cosine_sim : torch.Tensor
            Cosine similarity matrix of shape (N, N).
        labels : dict[str, torch.Tensor]
            Multi-hot positive label masks:
                - "ag": (N, N) bool tensor for antigen queries.
                - "ab": (N, N) bool tensor for antibody queries.
        grad_norm : float
            Gradient norm for the current batch.
        """
        self.total_samples += batch_size
        self.batch_count += 1
        self.total_loss += loss_val * batch_size  # Sum loss over samples

        logits_ag = logits["ag"].detach().cpu()
        logits_ab = logits["ab"].detach().cpu()
        cosine_sim = cosine_sim.detach().cpu()

        pred_ag = logits_ag.argmax(dim=1)
        pred_ab = logits_ab.argmax(dim=1)
        batch_indices = torch.arange(logits_ag.size(0))

        ag_labels = labels["ag"].detach().cpu()
        ab_labels = labels["ab"].detach().cpu()

        # Collect metrics
        # Accuracy
        self.correct_ag += int((ag_labels[batch_indices, pred_ag] == 1).sum().item())
        self.correct_ab += int((ab_labels[batch_indices, pred_ab] == 1).sum().item())

        # Recall@k, Precision@k
        for k in self.k_values:
            if batch_size >= k:
                # Recall
                hits_ag, hits_ab = self._evaluate_hits_at_k(
                    logits_ag, logits_ab, ag_labels, ab_labels, k_value=k
                )
                self.hits_at_k[k]["ag"] += hits_ag
                self.hits_at_k[k]["ab"] += hits_ab

                # Precision
                _, topk_ag = logits_ag.topk(k, dim=1)
                _, topk_ab = logits_ab.topk(k, dim=1)

                topk_ag_labels = ag_labels[batch_indices.unsqueeze(1), topk_ag]
                topk_ab_labels = ab_labels[batch_indices.unsqueeze(1), topk_ab]
                self.items_at_k[k]["ag"].append(topk_ag_labels)
                self.items_at_k[k]["ab"].append(topk_ab_labels)

        # EF contributions
        ## Accumulate ag->ab
        num, den = self._ef_batch_contrib(cosine_sim, ag_labels, self.ef_alpha)
        self._ef_num_ag2ab += num
        self._ef_den_ag2ab += den

        ## Accumulate ab->ag
        num, den = self._ef_batch_contrib(cosine_sim.t(), ab_labels, self.ef_alpha)
        self._ef_num_ab2ag += num
        self._ef_den_ab2ag += den

        # Take max among multi-positives for each sample
        self.total_logits_ag += (
            (logits_ag * ag_labels.float()).max(dim=1)[0].sum().item()
        )
        self.total_logits_ab += (
            (logits_ab * ab_labels.float()).max(dim=1)[0].sum().item()
        )
        self.total_cosine_sim += (
            (cosine_sim * ag_labels.float()).max(dim=1)[0].sum().item()
        )

        margin_max_neg_ag, margin_max_neg_ab = self._calculate_margin_max_neg_multipos(
            cosine_sim, ag_labels, ab_labels
        )

        # Update MRR
        self._accumulate_mrr_from_logits_labels(
            logits_q2c=logits_ag,
            labels_q2c=ag_labels,
            direction="ag2ab",
        )
        self._accumulate_mrr_from_logits_labels(
            logits_q2c=logits_ab,
            labels_q2c=ab_labels,
            direction="ab2ag",
        )

        self.total_margin_max_neg_ag += margin_max_neg_ag * batch_size
        self.total_margin_max_neg_ab += margin_max_neg_ab * batch_size

        self.total_grad_norm += grad_norm
        self.max_grad_norm = max(self.max_grad_norm, grad_norm)

    def _calculate_margin_max_neg(
        self, cosine_sim: torch.Tensor, labels: torch.Tensor
    ) -> tuple[float, float]:
        """Calculate margin for the standard CLIP case (1D labels).

        Parameters
        ----------
        cosine_sim : torch.Tensor
            Cosine similarity matrix of shape (N, N).
        labels : torch.Tensor
            1D tensor of positive class indices of shape (N,).

        Returns
        -------
        tuple[float, float]
            Average margin for antigen queries and antibody queries respectively.
        """
        batch_size = cosine_sim.size(0)
        batch_indices = torch.arange(batch_size, device=cosine_sim.device)

        # Convert 1D labels to 2D positive masks for unified processing
        ag_pos_mask = torch.zeros_like(cosine_sim, dtype=torch.bool)
        ab_pos_mask = torch.zeros_like(cosine_sim, dtype=torch.bool)
        ag_pos_mask[batch_indices, labels] = True  # Row-wise positives (ag->ab)
        ab_pos_mask[labels, batch_indices] = True  # Column-wise positives (ab->ag)

        return self._calculate_margin_unified(cosine_sim, ag_pos_mask, ab_pos_mask)

    def _calculate_margin_max_neg_multipos(
        self, cosine_sim: torch.Tensor, ag_labels: torch.Tensor, ab_labels: torch.Tensor
    ) -> tuple[float, float]:
        """Calculate margin for multi-positive case (2D label masks).

        Parameters
        ----------
        cosine_sim : torch.Tensor
            Cosine similarity matrix of shape (N, N).
        ag_labels : torch.Tensor
            Multi-hot positive mask for antigen queries of shape (N, N).
        ab_labels : torch.Tensor
            Multi-hot positive mask for antibody queries of shape (N, N).

        Returns
        -------
        tuple[float, float]
            Average margin for antigen queries and antibody queries respectively.
        """
        return self._calculate_margin_unified(
            cosine_sim, ag_labels.bool(), ab_labels.bool()
        )

    def _calculate_margin_unified(
        self,
        cosine_sim: torch.Tensor,
        ag_pos_mask: torch.Tensor,
        ab_pos_mask: torch.Tensor,
    ) -> tuple[float, float]:
        """Unified margin calculation for both 1D and 2D label cases.

        Parameters
        ----------
        cosine_sim : torch.Tensor
            Cosine similarity matrix of shape (N, N).
        ag_pos_mask : torch.Tensor
            Boolean mask of shape (N, N) where True indicates a positive ag→ab pair.
        ab_pos_mask : torch.Tensor
            Boolean mask of shape (N, N) where True indicates a positive ab→ag pair.

        Returns
        -------
        tuple[float, float]
            Average margin for antigen queries and antibody queries respectively.
        """
        # Calculate margins for antigen queries (row-wise)
        margins_ag = []
        for i in range(cosine_sim.size(0)):
            if ag_pos_mask[i].any():  # If there are positive pairs for this antigen
                pos_similarities = cosine_sim[i, ag_pos_mask[i]]
                # Best positive for this query
                max_pos = pos_similarities.max()
                # Hardest negative (exclude all positives)
                max_neg = cosine_sim[i].masked_fill(ag_pos_mask[i], -1e9).max()
                margin = max_pos - max_neg
                margins_ag.append(margin.item())

        # Calculate margins for antibody queries (column-wise)
        margins_ab = []
        for j in range(cosine_sim.size(1)):
            if ab_pos_mask[:, j].any():  # If there are positive pairs for this antibody
                pos_similarities = cosine_sim[ab_pos_mask[:, j], j]
                # Best positive for this query
                max_pos = pos_similarities.max()
                # Hardest negative (exclude all positives)
                max_neg = cosine_sim[:, j].masked_fill(ab_pos_mask[:, j], -1e9).max()
                margin = max_pos - max_neg
                margins_ab.append(margin.item())

        # Return average margins, or 0 if no valid margins
        avg_margin_ag = sum(margins_ag) / len(margins_ag) if margins_ag else 0.0
        avg_margin_ab = sum(margins_ab) / len(margins_ab) if margins_ab else 0.0

        return avg_margin_ag, avg_margin_ab

    def _evaluate_hits_at_k(
        self,
        logits_ag: torch.Tensor,
        logits_ab: torch.Tensor,
        ag_labels: torch.Tensor,
        ab_labels: torch.Tensor,
        k_value: int,
    ) -> tuple[float, float]:
        """Evaluate recall at a given k value.

        Parameters
        ----------
        logits_ag : torch.Tensor
            Logits for antigen-to-antibody queries of shape (N, N).
        logits_ab : torch.Tensor
            Logits for antibody-to-antigen queries of shape (N, N).
        ag_labels : torch.Tensor
            Multi-hot positive label mask for antigen queries of shape (N, N).
        ab_labels : torch.Tensor
            Multi-hot positive label mask for antibody queries of shape (N, N).
        k_value : int
            The value of k at which to compute recall (hits).

        Returns
        -------
        tuple[float, float]
            Number of antigen-query hits and antibody-query hits at k.
        """
        batch_indices = torch.arange(logits_ag.size(0)).unsqueeze(1)  # (batch_size, 1)

        _, topk_ag = logits_ag.topk(k_value, dim=1)  # (batch_size, k)
        _, topk_ab = logits_ab.topk(k_value, dim=1)  # (batch_size, k)

        topk_ag_labels = ag_labels[batch_indices, topk_ag]  # (batch_size, k)
        topk_ab_labels = ab_labels[batch_indices, topk_ab]  # (batch_size, k)

        # Recall@k: binary of whether any top-k item is relevant)
        hits_ag = (topk_ag_labels.sum(dim=1) > 0).float().sum().item()  # (batch_size,)
        hits_ab = (topk_ab_labels.sum(dim=1) > 0).float().sum().item()  # (batch_size,)

        return hits_ag, hits_ab

    @torch.no_grad()
    def _accumulate_mrr_from_logits_labels(
        self,
        logits_q2c: torch.Tensor,
        labels_q2c: torch.Tensor,
        *,
        direction: str,
    ) -> None:
        """Accumulate MRR components for queries -> candidates.

        Accumulate MRR components (sum of reciprocal ranks and valid counts)
        for a single direction: queries -> candidates.

        Assumption (ENFORCED):
        Every query must have at least one positive label.

        Parameters
        ----------
        logits_q2c : torch.Tensor
            Similarity scores of shape (n_queries, n_candidates); higher is better.
        labels_q2c : torch.Tensor
            Multi-hot labels of shape (n_queries, n_candidates), where 1 indicates
            a positive candidate.
        direction : str
            Either ``"ag2ab"`` or ``"ab2ag"``; used for bookkeeping.
        """
        if logits_q2c.shape != labels_q2c.shape:
            raise ValueError(
                f"MRR shape mismatch: logits {logits_q2c.shape} vs labels {labels_q2c.shape}"
            )

        # Sort candidates by descending score for each query
        ranked_idx = torch.argsort(logits_q2c, dim=1, descending=True)  # (nq, nc)
        ranked_labels = torch.gather(
            labels_q2c.to(torch.bool), dim=1, index=ranked_idx
        )  # (nq, nc)

        has_pos = ranked_labels.any(dim=1)  # (nq,)
        if not has_pos.all():
            bad_indices = (~has_pos).nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(
                f"MRR - Queries with no positives found at indices: {bad_indices}"
            )

        nq, nc = ranked_labels.shape

        # First positive rank per query (1-based)
        positions = torch.arange(1, nc + 1, device=logits_q2c.device).unsqueeze(
            0
        )  # (1, nc)
        pos_positions = torch.where(
            ranked_labels,
            positions,
            torch.full_like(positions, nc + 1),
        )
        first_rank = pos_positions.min(dim=1).values  # (nq,)

        # Reciprocal ranks
        rr = 1.0 / first_rank.to(torch.float32)

        if direction == "ag2ab":
            self.sum_rr_ag2ab += float(rr.sum().item())
            self.count_rr_ag2ab += int(nq)
            self.sum_rank_ag2ab += float(first_rank.sum().item())
            self.max_rank_ag2ab = max(self.max_rank_ag2ab, int(first_rank.max().item()))
        elif direction == "ab2ag":
            self.sum_rr_ab2ag += float(rr.sum().item())
            self.count_rr_ab2ag += int(nq)
            self.sum_rank_ab2ag += float(first_rank.sum().item())
            self.max_rank_ab2ag = max(self.max_rank_ab2ag, int(first_rank.max().item()))
        else:
            raise ValueError(f"Unknown MRR direction: {direction}")

    @staticmethod
    def _ef_batch_contrib(
        sim: torch.Tensor, pos_mask: torch.Tensor, alpha: float
    ) -> tuple[float, float]:
        """Compute micro-average Enrichment Factor contributions for a batch.

        Parameters
        ----------
        sim : torch.Tensor
            Similarity matrix of shape (B, B); higher scores are better.
        pos_mask : torch.Tensor
            Boolean positive mask of shape (B, B) where True indicates a positive
            pair for each query row.
        alpha : float
            Fraction of candidates considered as the top-k cutoff (e.g. 0.01).

        Returns
        -------
        tuple[float, float]
            ``(num, den)`` where ``num`` is the sum of positives retrieved in
            the top-k for each query and ``den`` is the sum of total positives
            per query.
        """
        # Ensure boolean
        pos_mask = pos_mask.to(dtype=torch.bool)

        batch_size = sim.shape[0]
        if batch_size == 0:
            return 0.0, 0.0

        k = int(math.ceil(alpha * batch_size))
        k = max(1, min(k, batch_size))

        # indices of top-k candidates for each query (row)
        topk_idx = torch.topk(sim, k=k, dim=1, largest=True).indices  # (B, k)

        # gather positives in top-k
        topk_pos = pos_mask.gather(dim=1, index=topk_idx)  # (B, k) bool
        a_i = topk_pos.sum(dim=1).to(dtype=torch.float32)  # (B,)

        # total positives per query (within batch candidate set)
        a_total_i = pos_mask.sum(dim=1).to(dtype=torch.float32)  # (B,)

        # skip queries with A_i == 0 (EF undefined)
        valid = a_total_i > 0
        num = float(a_i[valid].sum().item())
        den = float(a_total_i[valid].sum().item())

        return num, den

    def add_custom_metric(self, name: str, value: float) -> None:
        """Add a custom metric value for the current epoch.

        Parameters
        ----------
        name : str
            Name of the metric.
        value : float
            Value to record.
        """
        if name not in self.custom_metrics:
            self.custom_metrics[name] = []
        self.custom_metrics[name].append(value)

    def get_epoch_results(self) -> dict[str, float]:
        """Calculate and return final epoch metrics.

        Returns
        -------
        dict[str, float]
            Dictionary containing all calculated metrics.
        """
        if self.total_samples == 0:
            return self._get_empty_results()

        ef_ag2ab = (
            self._ef_num_ag2ab / (self.ef_alpha * self._ef_den_ag2ab)
            if self._ef_den_ag2ab > 0
            else 0.0
        )
        ef_ab2ag = (
            self._ef_num_ab2ag / (self.ef_alpha * self._ef_den_ab2ag)
            if self._ef_den_ab2ag > 0
            else 0.0
        )

        results: dict[str, float] = {
            # Core metrics
            "loss": self.total_loss / self.total_samples,
            "samples": float(self.total_samples),
            "batch_count": float(self.batch_count),
            # Accuracy metrics
            "pred_acc_ag": self.correct_ag / self.total_samples,
            "pred_acc_ab": self.correct_ab / self.total_samples,
            "pred_acc": (self.correct_ag + self.correct_ab) / (2 * self.total_samples),
            # Average logits (diagonal components)
            "avg_logits_ag": self.total_logits_ag / self.total_samples,
            "avg_logits_ab": self.total_logits_ab / self.total_samples,
            "avg_logits": (self.total_logits_ag + self.total_logits_ab)
            / (2 * self.total_samples),
            # Average cosine similarities
            "avg_cosine_sim": self.total_cosine_sim / self.total_samples,
            "avg_margin_max_neg_ag": self.total_margin_max_neg_ag / self.total_samples,
            "avg_margin_max_neg_ab": self.total_margin_max_neg_ab / self.total_samples,
            "avg_margin_max_neg": (
                self.total_margin_max_neg_ag + self.total_margin_max_neg_ab
            )
            / (2 * self.total_samples),
            "avg_grad_norm": (
                self.total_grad_norm / self.batch_count if self.batch_count > 0 else 0.0
            ),
            "max_grad_norm": self.max_grad_norm,
            # MRR metrics
            "mrr_ag2ab": self.sum_rr_ag2ab / max(self.count_rr_ag2ab, 1),
            "mrr_ab2ag": self.sum_rr_ab2ag / max(self.count_rr_ab2ag, 1),
            "mrr": (
                self.sum_rr_ag2ab / max(self.count_rr_ag2ab, 1)
                + self.sum_rr_ab2ag / max(self.count_rr_ab2ag, 1)
            )
            / 2.0,
            "mrr_avg_rank_ag2ab": self.sum_rank_ag2ab / max(self.count_rr_ag2ab, 1),
            "mrr_avg_rank_ab2ag": self.sum_rank_ab2ag / max(self.count_rr_ab2ag, 1),
            "mrr_max_rank_ag2ab": float(self.max_rank_ag2ab),
            "mrr_max_rank_ab2ag": float(self.max_rank_ab2ag),
            # Enrichment Factor (EF) metrics
            "ef1_ag2ab": ef_ag2ab,
            "ef1_ab2ag": ef_ab2ag,
            "ef1": (ef_ag2ab + ef_ab2ag) / 2.0,
        }

        # Add recall@k, precision@k metrics (averaged over all samples)
        for k in self.k_values:
            results[f"recall_at_{k}_ag"] = self.hits_at_k[k]["ag"] / self.total_samples
            results[f"recall_at_{k}_ab"] = self.hits_at_k[k]["ab"] / self.total_samples
            results[f"recall_at_{k}"] = (
                self.hits_at_k[k]["ag"] + self.hits_at_k[k]["ab"]
            ) / (2 * self.total_samples)

            if self.items_at_k[k]["ag"] and self.items_at_k[k]["ab"]:
                results[f"precision_at_{k}_ag"] = float(
                    (torch.cat(self.items_at_k[k]["ag"], dim=0).sum(dim=1) / k)
                    .float()
                    .mean()
                    .item()
                )
                results[f"precision_at_{k}_ab"] = float(
                    (torch.cat(self.items_at_k[k]["ab"], dim=0).sum(dim=1) / k)
                    .float()
                    .mean()
                    .item()
                )
                results[f"precision_at_{k}"] = float(
                    (
                        (
                            torch.cat(self.items_at_k[k]["ag"], dim=0).sum(dim=1)
                        + torch.cat(self.items_at_k[k]["ab"], dim=0).sum(dim=1)
                    )
                    / (2 * k)
                )
                .float()
                .mean()
                .item()
                )
            else:
                results[f"precision_at_{k}_ag"] = 0.0
                results[f"precision_at_{k}_ab"] = 0.0
                results[f"precision_at_{k}"] = 0.0

        # Add custom metrics (averaged)
        for metric_name, values in self.custom_metrics.items():
            if values:
                results[f"avg_{metric_name}"] = sum(values) / len(values)
                # results[f"total_{metric_name}"] = sum(values)

        return results

    def _get_empty_results(self) -> dict[str, float]:
        """Return empty results when no samples processed.

        Returns
        -------
        dict
            Dictionary containing empty metric values
        """
        return {
            "loss": 0.0,
            "samples": 0,
            "batch_count": 0,
            "pred_acc_ag": 0.0,
            "pred_acc_ab": 0.0,
            "pred_acc": 0.0,
            "avg_logits_ag": 0.0,
            "avg_logits_ab": 0.0,
            "avg_logits": 0.0,
            "avg_cosine_sim": 0.0,
            "avg_margin_max_neg_ag": 0.0,
            "avg_margin_max_neg_ab": 0.0,
            "avg_margin_max_neg": 0.0,
            "avg_grad_norm": 0.0,
            "max_grad_norm": 0.0,
            "mrr_ag2ab": 0.0,
            "mrr_ab2ag": 0.0,
            "mrr": 0.0,
            "mrr_avg_rank_ag2ab": 0.0,
            "mrr_avg_rank_ab2ag": 0.0,
            "mrr_max_rank_ag2ab": 0.0,
            "mrr_max_rank_ab2ag": 0.0,
        }

    def get_summary_string(self, phase: str) -> str:
        """Get a formatted summary string of current metrics.

        Parameters
        ----------
        phase : str
            Phase name for formatting.

        Returns
        -------
        str
            Formatted summary string.
        """
        results = self.get_epoch_results()
        phase_str = f"{phase} " if phase else ""

        return (
            f"{phase_str}Loss: {results['loss']:.4f}, "
            f"Acc_AG: {results['pred_acc_ag']:.4f}, "
            f"Acc_AB: {results['pred_acc_ab']:.4f}, "
            f"Samples: {results['samples']}"
        )


def save_results(
    metrics_dict: dict[str, list[dict[str, float]]], file_out: str
) -> None:
    """Save epoch metrics to a CSV file.

    Parameters
    ----------
    metrics_dict : dict
        Dictionary with metric dicts per phase per epoch
            - "train": [ {metric_dict_epoch1}, {metric_dict_epoch2}, ... ]
            - "val": [ {metric_dict_epoch1}, {metric_dict_epoch2}, ... ]
            - "test": [ {metric_dict_epoch1}, {metric_dict_epoch2}, ...
    file_out : str
        Output CSV filename
    """
    rows = []
    for phase, metrics_list in metrics_dict.items():
        for entry in metrics_list:
            row = {"phase": phase, **entry}
            rows.append(row)

    if not rows:
        print("No metrics to save.")
        return

    header = sorted(rows[0].keys())
    try:
        with open(file_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Metrics saved to {file_out}")
    except Exception as e:
        print(f"Error saving metrics: {e}")
