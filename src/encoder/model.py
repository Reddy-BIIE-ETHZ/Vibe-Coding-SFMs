"""
Model definitions and architectures.

This module defines the model architectures used for contrastive learning between
antigen and antibody embeddings based on CLIP architecture
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as f
from omegaconf import DictConfig


_EMBED_DIMS: dict[str, int] = {
    "esm2": 1280,
    "antiberty": 512,
    "dnabert2": 768,
    "molformer": 768,
}


def _get_embed_dim(model_name: str) -> int:
    """Look up embedding dimension for a pre-trained encoder backbone."""
    if model_name not in _EMBED_DIMS:
        raise ValueError(
            f"Unknown encoder model '{model_name}'. "
            f"Supported: {list(_EMBED_DIMS.keys())}"
        )
    return _EMBED_DIMS[model_name]


def build_encoder_model(cfg: DictConfig) -> nn.Module:
    """Load the appropriate model based on the provided arguments.

    Parameters
    ----------
    cfg : DictConfig
        Arguments containing model configuration

    Returns
    -------
    nn.Module
        The constructed model.
    """
    model_cfg = cfg.model.encoder
    reduce_embeddings = cfg.train.encoder.reduce_embeddings

    return CALMEncoder(model_cfg, reduce_embeddings=reduce_embeddings)


class CALMEncoder(nn.Module):
    """CALM (Contrastive Antigen-antibody Learning Model) Encoder.

    This class implements CLIP-style contrastive learning between antigen and
    antibody embeddings. Supports different encoder architectures (FFN or Transformer).

    Parameters
    ----------
    cfg : DictConfig
        Configuration containing model parameters
    encoder_type : str
        Type of encoder architecture: "ffn" for feed-forward networks or
        "transformer" for transformer networks
    reduce_embeddings : bool
        Whether to use reduced embeddings (affects max sequence length for transformers)
    """

    def __init__(
        self,
        cfg: DictConfig,
        reduce_embeddings: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.encoder_type = cfg.encoder_type
        self.reduce_embeddings = reduce_embeddings

        # Per-source learnable temperature (mhcSFM v2 Phase 1.3, 2026-05-01).
        # cfg.n_sources controls how many independent temperatures the model
        # learns. Default is 1, which preserves v1 behavior exactly: the
        # underlying parameter shape becomes (1,) instead of (), but the
        # forward math is identical (cos * scalar). Domains with heterogeneous
        # data sources (e.g., binding-affinity + eluted-ligand = 2 sources)
        # set n_sources accordingly in their model config and pass per-pair
        # source_idx into forward().
        self.n_sources = getattr(cfg, "n_sources", 1)
        self.logit_scale = nn.Parameter(
            torch.full((self.n_sources,), math.log(1 / cfg.tau))
        )
        self.max_scale = cfg.max_scale

        # Phase 1.6 fix (mhcSFM v2, 2026-05-02): optionally freeze the
        # learnable temperature(s). On highly imbalanced source distributions
        # (e.g., mhcSFM v2.1 has 5% BA + 95% EL), the dominant source's tau
        # gets ~20x more gradient updates per epoch than the minority source's,
        # producing runaway divergence (tau_EL drifted 0.07 -> 1.03 in epoch 0
        # while tau_BA only reached 0.24). When freeze_temperature=True, the
        # logit_scale parameter is created but excluded from optimizer updates.
        # Default False preserves the standard behavior in which every SFM
        # learns its temperature as before. NetMHCpan, the field reference
        # for peptide-MHC, also uses a fixed temperature.
        if getattr(cfg, "freeze_temperature", False):
            self.logit_scale.requires_grad = False

        # Build encoders based on architecture type
        if self.encoder_type == "ffn":
            self.encoder_ag, self.encoder_ab = self._build_ffn_encoders(cfg)
        elif self.encoder_type == "transformer":
            self.encoder_ag, self.encoder_ab = self._build_transformer_encoders(
                cfg, reduce_embeddings
            )
        else:
            raise ValueError(
                f"Unsupported encoder_type: {self.encoder_type}. Use 'ffn' or 'transformer'."
            )

        # Symmetric-encoder option (any Siamese-style domain).
        # When cfg.symmetric is True, tie encoder_ab to encoder_ag so the same
        # weights encode both sides of every pair. Appropriate for undirected
        # pair data (e.g., HuRI human PPI) where (A, B) and (B, A) are the
        # same biological event. Also ties the attention pooling heads when
        # FFN + pooling="attn". Default is False — unchanged behavior for
        # every SFM in this release (tSFM, crisprSFM, dtSFM, eSFM, mhcSFM, mir-SFM).
        if getattr(cfg, "symmetric", False):
            self.encoder_ab = self.encoder_ag
            if self.encoder_type == "ffn" and getattr(self, "pooling", None) == "attn":
                self.attention_ab = self.attention_ag

    def _build_ffn_encoders(self, cfg: DictConfig) -> tuple[nn.Module, nn.Module]:
        """Build feed-forward network encoders for antigen and antibody.

        Parameters
        ----------
        cfg : DictConfig
            Configuration containing model parameters

        Returns
        -------
        tuple[nn.Module, nn.Module]
            Antigen and antibody FFN encoders
        """
        # Determine input dimensions
        dim_input_ag = _get_embed_dim(cfg.model_ag)
        dim_input_ab = _get_embed_dim(cfg.model_ab)

        include_linear_bias = cfg.include_linear_bias

        encoder_ag = nn.Sequential(
            nn.Linear(dim_input_ag, cfg.d_ff, bias=include_linear_bias),
            nn.ReLU(),
            nn.Linear(cfg.d_ff, cfg.d_model, bias=include_linear_bias),
        )
        encoder_ab = nn.Sequential(
            nn.Linear(dim_input_ab, cfg.d_ff, bias=include_linear_bias),
            nn.ReLU(),
            nn.Linear(cfg.d_ff, cfg.d_model, bias=include_linear_bias),
        )

        # Initialize FFN encoders
        def init_linear(m: nn.Module) -> None:
            """Initialize weights of a linear layer with normal distribution."""
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        encoder_ag.apply(init_linear)
        encoder_ab.apply(init_linear)

        # Store pooling configuration for FFN
        self.pooling = cfg.pooling
        if self.pooling == "attn":
            self.attention_ag = nn.Linear(cfg.d_model, 1, bias=include_linear_bias)
            self.attention_ab = nn.Linear(cfg.d_model, 1, bias=include_linear_bias)
            init_linear(self.attention_ag)
            init_linear(self.attention_ab)

        return encoder_ag, encoder_ab

    def _build_transformer_encoders(
        self, cfg: DictConfig, reduce_embeddings: bool
    ) -> tuple[nn.Module, nn.Module]:
        """Build transformer encoders for antigen and antibody.

        Parameters
        ----------
        cfg : DictConfig
            Configuration containing model parameters
        reduce_embeddings : bool
            Whether to use reduced embeddings (affects max sequence length)

        Returns
        -------
        tuple[nn.Module, nn.Module]
            Antigen and antibody transformer encoders
        """
        # Determine input dimensions
        dim_input_ag = _get_embed_dim(cfg.model_ag)
        dim_input_ab = _get_embed_dim(cfg.model_ab)

        # Store pooling configuration for transformers
        self.pooling = cfg.pooling

        # Determine max lengths based on reduce_embeddings and pooling
        # Allow config overrides for domain adaptation (tSFM, etc.)
        if hasattr(cfg, "max_length_ag") and cfg.max_length_ag is not None:
            max_length_ag = cfg.max_length_ag + (self.pooling == "cls")
        elif reduce_embeddings:
            max_length_ag = 162 + (self.pooling == "cls")
        else:
            max_length_ag = 550 + (self.pooling == "cls")

        if hasattr(cfg, "max_length_ab") and cfg.max_length_ab is not None:
            max_length_ab = cfg.max_length_ab + (self.pooling == "cls")
        elif reduce_embeddings:
            max_length_ab = 184 + (self.pooling == "cls")
        else:
            max_length_ab = 282 + (self.pooling == "cls")

        # Build transformer encoders
        encoder_ag = SequenceTransformerEncoder(
            vocab_size=-1,
            context_length=max_length_ag,  # +1 if cls token
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
            use_token_embedding=False,
            input_embed_dim=dim_input_ag,
            pooling=self.pooling,  # "eos" or "cls"
            include_projection_final=cfg.include_projection_final,
        )
        encoder_ab = SequenceTransformerEncoder(
            vocab_size=-1,
            context_length=max_length_ab,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
            use_token_embedding=False,
            input_embed_dim=dim_input_ab,
            pooling=self.pooling,  # "eos" or "cls"
            include_projection_final=cfg.include_projection_final,
        )

        return encoder_ag, encoder_ab

    def masked_mean_pooling(
        self, embeddings: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply masked mean pooling along the sequence dimension.

        Parameters
        ----------
        embeddings : torch.Tensor
            Input embeddings of shape (N, L, D)
        mask : torch.Tensor, optional
            Boolean mask of shape (N, L) where True indicates valid positions

        Returns
        -------
        torch.Tensor
            Pooled features of shape (N, D)
        """
        if mask is not None:
            # Expand mask to match embedding dimensions: (N, L) -> (N, L, 1)
            mask_expanded = mask.unsqueeze(-1).float()
            # Masked sum and count for mean calculation
            pooled = (embeddings * mask_expanded).sum(dim=1) / mask_expanded.sum(
                dim=1
            ).clamp(min=1)
        else:
            # Check if all elements in the feature dimension are zero: (N, L, D) -> (N, L)
            non_zero_mask = (embeddings != 0).any(dim=-1)  # (N, L)
            # Expand mask to match embedding dimensions: (N, L) -> (N, L, 1)
            non_zero_mask_expanded = non_zero_mask.unsqueeze(-1).float()
            # Masked sum and count for mean calculation, excluding zero elements
            pooled = (embeddings * non_zero_mask_expanded).sum(
                dim=1
            ) / non_zero_mask_expanded.sum(dim=1).clamp(min=1)
        return pooled

    def masked_attn_pooling(
        self,
        embeddings: torch.Tensor,
        attention_layer: nn.Module,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply masked attention pooling along the sequence dimension.

        Parameters
        ----------
        embeddings : torch.Tensor
            Input embeddings of shape (N, L, D)
        attention_layer : nn.Module
            Attention layer for computing attention scores
        mask : torch.Tensor, optional
            Boolean mask of shape (N, L) where True indicates valid positions

        Returns
        -------
        torch.Tensor
            Pooled features of shape (N, D)
        """
        # Compute attention scores: (N, L, D) -> (N, L, 1) -> (N, L)
        attention_scores = attention_layer(embeddings).squeeze(-1)  # (N, L)

        if mask is not None:
            # Apply mask: set masked positions to large negative value
            attention_scores = attention_scores.masked_fill(~mask, -1e9)

        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=1)  # (N, L)

        # Apply attention weights to embeddings
        # attention_weights: (N, L) -> (N, L, 1) for broadcasting
        attention_weights_expanded = attention_weights.unsqueeze(-1)  # (N, L, 1)

        # Weighted sum: (N, L, D) * (N, L, 1) -> (N, L, D) -> (N, D)
        pooled = (embeddings * attention_weights_expanded).sum(dim=1)  # (N, D)

        return pooled

    def forward(
        self,
        ag_emb: torch.Tensor,
        ab_emb: torch.Tensor,
        ag_mask: torch.Tensor,
        ab_mask: torch.Tensor,
        source_idx: torch.Tensor | None = None,
    ) -> tuple[
        dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict[str, torch.Tensor]
    ]:
        """Forward pass for CLIP-style contrastive learning.

        Parameters
        ----------
        ag_emb : torch.Tensor
            Antigen embeddings of shape (N, L, D)
        ab_emb : torch.Tensor
            Antibody embeddings of shape (N, L, D)
        ag_mask : torch.Tensor
            Antigen attention masks
        ab_mask : torch.Tensor
            Antibody attention masks
        source_idx : torch.Tensor, optional
            Per-pair source index (long tensor, shape (N,)). When provided
            and the model has n_sources > 1, the forward uses the anchor's
            source temperature for that row of the (N, N) logits matrix
            (anchor's-source convention). When source_idx is None or n_sources == 1,
            the forward uses a single scalar temperature, which is the v1
            behavior exactly. Default: None (back-compat).

        Returns
        -------
        tuple
            A tuple containing:
            - logits: Dict with keys 'ag' and 'ab' for model outputs
            - cosine_sim: Cosine similarity matrix
            - logit_scale: Learnable scaling factor for logits (scalar; mean
              of per-pair scales when per-source path is active, for logging)
            - outputs: Dict containing features and projections
        """
        if self.encoder_type == "ffn":
            # FFN encoder path
            ag_emb_proj = self.encoder_ag(ag_emb)  # (N, L, D)
            ab_emb_proj = self.encoder_ab(ab_emb)  # (N, L, D)

            ag_emb_proj = ag_emb_proj * ag_mask.unsqueeze(-1)  # (N, L, D) * (N, L, 1)
            ab_emb_proj = ab_emb_proj * ab_mask.unsqueeze(-1)  # (N, L, D) * (N, L, 1)

            # length-wise pooling with mask support
            if self.pooling == "attn":
                ag_features = self.masked_attn_pooling(
                    ag_emb_proj, self.attention_ag, mask=ag_mask
                )
                ab_features = self.masked_attn_pooling(
                    ab_emb_proj, self.attention_ab, mask=ab_mask
                )
            else:
                ag_features = self.masked_mean_pooling(ag_emb_proj, mask=ag_mask)
                ab_features = self.masked_mean_pooling(ab_emb_proj, mask=ab_mask)

        elif self.encoder_type == "transformer":
            # Transformer encoder path - returns both hidden states and pooled features
            ag_emb_proj, ag_features = self.encoder_ag(
                input_ids=None,
                inputs_embeds=ag_emb,
                attention_mask=ag_mask,
                pooling=self.pooling,
            )
            ab_emb_proj, ab_features = self.encoder_ab(
                input_ids=None,
                inputs_embeds=ab_emb,
                attention_mask=ab_mask,
                pooling=self.pooling,
            )

        # L2 normalization
        ag_features = f.normalize(ag_features, p=2, dim=-1)
        ab_features = f.normalize(ab_features, p=2, dim=-1)

        # cosine similarity & scaling
        cosine_sim = ag_features @ ab_features.t()
        # Per-source temperature (mhcSFM v2 Phase 1.3, design lock D6).
        # Three branches:
        #   - Per-source path: source_idx provided AND n_sources > 1.
        #     Scale row i of cosine_sim by exp(logit_scale[source_idx[i]]).
        #     Anchor's-source convention: the anchor (whichever side you
        #     view as anchor) determines the temperature for that row.
        #   - Back-compat scalar path: n_sources == 1. logit_scale has
        #     shape (1,) but exp().clamp() broadcasts cleanly against
        #     cosine_sim (B, B). Index [0] returns the 0-dim tensor so
        #     downstream .item() calls keep working.
        #   - Mean fallback: n_sources > 1 but source_idx not provided
        #     (e.g., during eval where source label may not be present).
        #     Use the mean of per-source scales as a representative value.
        if source_idx is not None and self.n_sources > 1:
            per_pair_scale = (
                self.logit_scale[source_idx].exp().clamp(max=self.max_scale)
            )  # (B,)
            logits_per_ag = cosine_sim * per_pair_scale.unsqueeze(1)  # (B,1)*(B,B)
            logits_per_ab = cosine_sim.t() * per_pair_scale.unsqueeze(1)
            logit_scale = per_pair_scale.mean()  # scalar for logging
        else:
            if self.n_sources == 1:
                logit_scale = self.logit_scale[0].exp().clamp(max=self.max_scale)
            else:
                logit_scale = (
                    self.logit_scale.exp().clamp(max=self.max_scale).mean()
                )
            logits_per_ag = cosine_sim * logit_scale
            logits_per_ab = logits_per_ag.t()

        logits = {"ag": logits_per_ag, "ab": logits_per_ab}
        outputs = {
            "features_ag": ag_features,
            "features_ab": ab_features,
            "projections_ag": ag_emb_proj,
            "projections_ab": ab_emb_proj,
        }
        return logits, cosine_sim, logit_scale, outputs


class SequenceTransformerEncoder(nn.Module):
    """Sequence Transformer Encoder.

    Implements a transformer-based encoder with support for token embeddings,
    positional embeddings, and various pooling strategies (CLS, EOS, or mean).

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary for token embeddings.
    context_length : int, optional
        Maximum sequence length, by default 77.
    d_model : int, optional
        Dimensionality of the model, by default 512.
    n_layers : int, optional
        Number of transformer encoder layers, by default 12.
    n_heads : int, optional
        Number of attention heads, by default 8.
    d_ff : int, optional
        Dimensionality of the feedforward network, by default 2048.
    dropout : float, optional
        Dropout rate, by default 0.1.
    use_token_embedding : bool, optional
        Whether to use token embeddings, by default True.
    input_embed_dim : int, optional
        Dimensionality of input embeddings from an external model, by default None.
    pooling : str, optional
        Pooling strategy (``'eos'``, ``'cls'``, or ``'mean'``), by default ``'eos'``.
    include_projection_final : bool, optional
        Whether to include a final linear projection layer, by default False.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int = 77,
        d_model: int = 512,
        n_layers: int = 12,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        use_token_embedding: bool = True,
        input_embed_dim: int | None = None,  # e.g. 1280 from a pretrained model
        pooling: str = "eos",  # "eos" or "cls"
        include_projection_final: bool = False,
    ):
        """Initialize the SequenceTransformerEncoder.

        Parameters
        ----------
        vocab_size : int
            Size of the vocabulary for token embeddings.
        context_length : int, optional
            Maximum sequence length, by default 77.
        d_model : int, optional
            Dimensionality of the model, by default 512.
        n_layers : int, optional
            Number of transformer encoder layers, by default 12.
        n_heads : int, optional
            Number of attention heads, by default 8.
        d_ff : int, optional
            Dimensionality of the feedforward network, by default 2048.
        dropout : float, optional
            Dropout rate, by default 0.1.
        use_token_embedding : bool, optional
            Whether to use token embeddings, by default True.
        input_embed_dim : int, optional
            Dimensionality of input embeddings, by default None.
        pooling : str, optional
            Pooling strategy ('eos', 'cls', or 'mean'), by default 'eos'.
        include_projection_final : bool, optional
            Whether to include the final projection layer, by default True.
        """
        super().__init__()
        if pooling not in ("eos", "cls", "mean"):
            raise ValueError("pooling must be 'eos', 'cls', or 'mean'")

        self.context_length = context_length
        self.d_model = d_model
        self.use_token_embedding = use_token_embedding
        self.pooling = pooling
        self.include_projection_final = include_projection_final

        if use_token_embedding:
            self.token_embedding: nn.Embedding | None = nn.Embedding(
                vocab_size, d_model
            )
        else:
            self.token_embedding = None  # expect external inputs_embeds

        # external embedding dim (e.g. 1280) → project to d_model (e.g. 512)
        self.input_embed_dim = input_embed_dim
        if self.input_embed_dim is not None and self.input_embed_dim != d_model:
            self.input_proj: nn.Linear | None = nn.Linear(self.input_embed_dim, d_model)
        else:
            self.input_proj = None

        # learned positional embeddings (for max length including potential CLS)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, d_model))

        # learned CLS token (used only if pooling == "cls")
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.ln_final = nn.LayerNorm(d_model)
        if self.include_projection_final:
            self.projection_final = nn.Parameter(torch.empty(d_model, d_model))

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize the weights of the model components.

        Uses normal initialization for embeddings and the final projection,
        Xavier uniform for attention and feedforward weights, and zeros/ones
        for biases/LayerNorm parameters.
        """
        if self.token_embedding is not None:
            nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if self.pooling == "cls":
            nn.init.normal_(self.cls_token, std=0.02)

        for layer in self.transformer.layers:
            # FFN
            nn.init.xavier_uniform_(layer.linear1.weight)
            if layer.linear1.bias is not None:
                nn.init.zeros_(layer.linear1.bias)

            nn.init.xavier_uniform_(layer.linear2.weight)
            if layer.linear2.bias is not None:
                nn.init.zeros_(layer.linear2.bias)

            # Self-attention
            attn = layer.self_attn

            if hasattr(attn, "in_proj_weight") and attn.in_proj_weight is not None:
                nn.init.xavier_uniform_(attn.in_proj_weight)
                if attn.in_proj_bias is not None:
                    nn.init.zeros_(attn.in_proj_bias)
            else:
                # Separate Q/K/V projection weights (rare in encoder layers, but possible)
                nn.init.xavier_uniform_(attn.q_proj_weight)
                nn.init.xavier_uniform_(attn.k_proj_weight)
                nn.init.xavier_uniform_(attn.v_proj_weight)
                if attn.in_proj_bias is not None:
                    nn.init.zeros_(attn.in_proj_bias)

            nn.init.xavier_uniform_(attn.out_proj.weight)
            if attn.out_proj.bias is not None:
                nn.init.zeros_(attn.out_proj.bias)

            # LayerNorm(s)
            for name in ("norm1", "norm2"):
                if hasattr(layer, name):
                    norm = getattr(layer, name)
                    nn.init.ones_(norm.weight)
                    nn.init.zeros_(norm.bias)

        if self.include_projection_final:
            nn.init.normal_(self.projection_final, std=0.02)

    def masked_mean_pooling(
        self, embeddings: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply masked mean pooling along the sequence dimension.

        Parameters
        ----------
        embeddings : torch.Tensor
            Input embeddings of shape (N, L, D).
        mask : torch.Tensor, optional
            Boolean mask of shape (N, L) where True indicates valid positions.

        Returns
        -------
        torch.Tensor
            Pooled features of shape (N, D).
        """
        if mask is not None:
            # Expand mask to match embedding dimensions: (N, L) -> (N, L, 1)
            mask_expanded = mask.unsqueeze(-1).float()
            # Masked sum and count for mean calculation
            pooled = (embeddings * mask_expanded).sum(dim=1) / mask_expanded.sum(
                dim=1
            ).clamp(min=1)
        else:
            # Check if all elements in the feature dimension are zero: (N, L, D) -> (N, L)
            non_zero_mask = (embeddings != 0).any(dim=-1)  # (N, L)
            # Expand mask to match embedding dimensions: (N, L) -> (N, L, 1)
            non_zero_mask_expanded = non_zero_mask.unsqueeze(-1).float()
            # Masked sum and count for mean calculation, excluding zero elements
            pooled = (embeddings * non_zero_mask_expanded).sum(
                dim=1
            ) / non_zero_mask_expanded.sum(dim=1).clamp(min=1)
        return pooled

    def forward(
        self,
        input_ids: torch.Tensor | None = None,  # (B, L)
        inputs_embeds: torch.Tensor | None = None,  # (B, L, input_embed_dim or d_model)
        attention_mask: torch.Tensor | None = None,  # (B, L), 1 = valid, 0 = pad
        eos_indices: torch.Tensor | None = None,  # (B,) if pooling="eos"
        pooling: str | None = None,  # override default pooling if not None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the SequenceTransformerEncoder.

        Parameters
        ----------
        input_ids : torch.Tensor, optional
            Input token IDs of shape (B, L), by default None.
        inputs_embeds : torch.Tensor, optional
            Input embeddings of shape (B, L, input_embed_dim or d_model), by default None.
        attention_mask : torch.Tensor, optional
            Attention mask of shape (B, L), where 1 indicates valid tokens and 0 indicates padding, by default None.
        eos_indices : torch.Tensor, optional
            End-of-sequence indices of shape (B,), used if pooling="eos", by default None.
        pooling : str, optional
            Pooling strategy to override the default, by default None.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            hidden_states : torch.Tensor
                Hidden states of shape (B, L_total, d_model).
            pooled : torch.Tensor
                Pooled output of shape (B, d_model).
        """
        # choose pooling mode
        if pooling is None:
            pooling = self.pooling

        use_cls = pooling == "cls"

        # input handling
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of `input_ids` or `inputs_embeds`.")

        if inputs_embeds is not None:
            x = inputs_embeds  # (B, L, embedding_dim)
            batch_size, batch_length, embedding_dim = x.shape

            # infer / check external dim
            if self.input_embed_dim is None:
                if embedding_dim != self.d_model and self.input_proj is None:
                    raise ValueError(
                        f"inputs_embeds last dim {embedding_dim} != d_model {self.d_model}, "
                        "and input_embed_dim is None. Set input_embed_dim in __init__ "
                        "or project externally."
                    )
            else:
                if embedding_dim != self.input_embed_dim:
                    raise ValueError(
                        f"inputs_embeds last dim {embedding_dim} != input_embed_dim {self.input_embed_dim}"
                    )

            if self.input_proj is not None:
                x = self.input_proj(x)  # (B, L, d_model)

        else:
            if not self.use_token_embedding or self.token_embedding is None:
                raise ValueError(
                    "use_token_embedding=False but `input_ids` was provided. "
                    "Either enable token embeddings or pass `inputs_embeds`."
                )
            x = self.token_embedding(input_ids)  # (B, L, d_model)
            if input_ids is None:
                raise ValueError("input_ids is None")
            batch_size, batch_length = input_ids.shape

        # build attention_mask if missing: assume all tokens valid
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, batch_length, device=x.device, dtype=torch.long
            )
        # If using CLS pooling, prepend CLS token + mask
        if use_cls:
            # cls_token: (1, 1, d_model) -> expand to batch
            cls_tok = self.cls_token.expand(
                batch_size, 1, self.d_model
            )  # (B, 1, d_model)
            x = torch.cat([cls_tok, x], dim=1)  # (B, L+1, d_model)

            cls_mask = torch.ones(
                batch_size, 1, device=attention_mask.device, dtype=attention_mask.dtype
            )
            attention_mask = torch.cat([cls_mask, attention_mask], dim=1)  # (B, L+1)

        # after optional CLS addition
        batch_size, batch_length_final, _ = x.shape

        if batch_length_final > self.context_length:
            raise ValueError(
                f"Sequence length {batch_length_final} exceeds context_length {self.context_length}"
            )

        # add positional embeddings
        pos = self.positional_embedding[:batch_length_final]  # (L_total, d_model)
        x = x + pos.unsqueeze(0)  # (B, L_total, d_model)

        # PyTorch expects True where positions should be masked (ignored)
        # Our convention: attention_mask == 1 -> keep, 0 -> pad
        # So we need to convert the mask
        src_key_padding_mask = attention_mask == 0  # (B, L_total), bool

        # transformer encoder (bidirectional, with padding mask)
        hidden_states = self.transformer(
            x,
            src_key_padding_mask=src_key_padding_mask,
        )  # (B, L_total, d_model)

        hidden_states = self.ln_final(hidden_states)
        if self.include_projection_final:
            hidden_states = hidden_states @ self.projection_final

        # pooling
        if pooling == "cls":
            # first position is CLS
            pooled = hidden_states[:, 0, :]  # (B, d_model)

        elif pooling == "eos":
            # no CLS added in this branch: L_total == original L
            if eos_indices is None:
                # infer EOS as last valid (mask==1) position per sequence
                lengths = attention_mask.sum(dim=1)  # (B,)
                # avoid negative indices if any sequence is all-pad (clamp to 0)
                eos_indices = (lengths - 1).clamp(min=0)

            pooled = hidden_states[
                torch.arange(batch_size, device=hidden_states.device), eos_indices
            ]  # (B, d_model)

        elif pooling == "mean":
            pooled = self.masked_mean_pooling(
                hidden_states, attention_mask
            )  # (B, d_model)

        return hidden_states, pooled
