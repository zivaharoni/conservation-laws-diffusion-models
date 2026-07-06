"""Regression and sequence models."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .transformer_backbone import (
    DiTTimestepEmbedder,
    SharedTransformerBackbone,
    TransformerModelOutput,
    apply_channel_tilt,
)


def _init_embedding(embedding: nn.Embedding) -> None:
    with torch.no_grad():
        nn.init.xavier_uniform_(embedding.weight)


def _init_embedding_normal(embedding: nn.Embedding, mean: float = 0.0, std: float = 1.0) -> None:
    with torch.no_grad():
        nn.init.normal_(embedding.weight, mean=mean, std=std)


def posterior_mean_from_logits(logits: torch.Tensor, x_values: torch.Tensor, topk: int | None = None) -> torch.Tensor:
    if topk is not None:
        alphabet_size = int(logits.shape[-1])
        k = max(1, min(int(topk), alphabet_size))
        if k < alphabet_size:
            topk_logits, topk_idx = torch.topk(logits, k=k, dim=-1)
            topk_probs = torch.softmax(topk_logits, dim=-1)
            topk_x = x_values[topk_idx]
            return torch.sum(topk_probs.unsqueeze(-1) * topk_x, dim=-2)
    probs = torch.softmax(logits, dim=-1)
    return torch.matmul(probs, x_values)


class DiscretePriorGaussianRegressionModelEmbedding(nn.Module):
    """IID regression model with learnable constellation embeddings."""

    def __init__(
        self,
        alphabet_size: int = 2,
        embed_hidden: int = 32,
        embed_dim: int = 32,
        **_: object,
    ) -> None:
        super().__init__()
        del embed_hidden

        self.embed_dim = embed_dim
        self.alphabet_size = alphabet_size
        self.x_values = nn.Embedding(alphabet_size, embed_dim)
        self.error_proj = nn.Linear(alphabet_size, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.prob_proj = nn.Linear(embed_dim, alphabet_size)

        _init_embedding(self.x_values)

    def forward(self, y: torch.Tensor, snr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        alphabet, d = self.x_values.weight.shape
        snr = torch.clamp(snr, min=1e-12).to(dtype=y.dtype, device=y.device)
        y_expanded = y.view(y.shape[0], 1, -1).expand(y.shape[0], alphabet, d)
        x_values = self.x_values.weight.to(dtype=y.dtype, device=y.device).view(1, alphabet, d).expand_as(y_expanded)
        error = d / 2 * torch.log(snr) - 0.5 * snr * ((y_expanded - x_values) ** 2).sum(dim=-1)
        error_emb = self.error_proj(error)

        emb_hat = self.out_proj(error_emb)
        logits_ext = self.prob_proj(error_emb)
        logits = apply_channel_tilt(logits_ext, error)
        return emb_hat, logits_ext, logits


class BidirectionalMaskedTokenModel(nn.Module):
    """Bidirectional self-prediction model for masked discrete sequences."""

    def __init__(
        self,
        *,
        alphabet_size: int,
        embed_dim: int,
        n_transformer_layers: int,
        n_heads: int,
        attention_dropout: float,
        max_seq_len: int,
    ) -> None:
        super().__init__()

        self.alphabet_size = int(alphabet_size)
        self.input_alphabet_size = self.alphabet_size + 1
        self.mask_token_id = self.alphabet_size
        self.embed_dim = int(embed_dim)
        self.n_transformer_layers = max(1, int(n_transformer_layers))
        self.n_heads = int(n_heads)
        self.max_seq_len = int(max_seq_len)
        self.attention_dropout = float(attention_dropout)

        self.x_values = nn.Embedding(self.input_alphabet_size, self.embed_dim)
        self.backbone = SharedTransformerBackbone(
            hidden_size=self.embed_dim,
            n_layers=self.n_transformer_layers,
            n_heads=self.n_heads,
            ffn_hidden_size=4 * self.embed_dim,
            dropout=self.attention_dropout,
            attention_mode="bidirectional",
        )
        self.prob_proj = nn.Linear(self.embed_dim, self.alphabet_size)

        _init_embedding_normal(self.x_values)

    def transformer_blocks(self):
        return self.backbone.transformer_blocks()

    def forward_outputs(
        self,
        tokens: torch.Tensor,
        *,
        t_embed: torch.Tensor | None = None,
        channel_tilt: torch.Tensor | None = None,
    ) -> TransformerModelOutput:
        if tokens.dim() != 2:
            raise ValueError(f"Expected tokens to be [B, S], got {tuple(tokens.shape)}")
        _, seq_len = tokens.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")
        tokens_t = tokens.long()
        hidden = self.backbone(self.x_values(tokens_t), t_embed=t_embed)
        logits_ext = self.prob_proj(hidden)
        return TransformerModelOutput(hidden=hidden, logits_ext=logits_ext, logits=apply_channel_tilt(logits_ext, channel_tilt))

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        t_embed: torch.Tensor | None = None,
        channel_tilt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_outputs(tokens, t_embed=t_embed, channel_tilt=channel_tilt).logits


class DiscretePriorGaussianRegressionModelEmbeddingSequence(nn.Module):
    """Sequence regression model for first-order Markov sources."""

    def __init__(
        self,
        alphabet_size: int = 2,
        embed_hidden: int = 32,
        embed_dim: int = 32,
        n_transformer_layers: int = 3,
        attention_dropout: float = 0.0,
        n_heads: int | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__()
        del embed_hidden

        self.embed_dim = int(embed_dim)
        self.alphabet_size = int(alphabet_size)
        self.n_transformer_layers = max(2, int(n_transformer_layers))
        self.attention_dropout = float(attention_dropout)
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError(f"attention_dropout must be in [0, 1), got {self.attention_dropout}")
        self.max_seq_len = int(kwargs.get("max_seq_len", 256))
        self.n_heads = int(n_heads) if n_heads is not None else (4 if self.embed_dim % 4 == 0 else 1)
        if self.embed_dim % self.n_heads != 0:
            raise ValueError(f"embed_dim={self.embed_dim} must be divisible by n_heads={self.n_heads}")
        if (self.embed_dim // self.n_heads) % 2 != 0:
            raise ValueError("RoPE requires embed_dim / n_heads to be even.")

        self.x_values = nn.Embedding(self.alphabet_size, self.embed_dim)
        self.error_proj = nn.Sequential(
            nn.Linear(self.alphabet_size, 4 * self.embed_dim),
            nn.GELU(),
            nn.Linear(4 * self.embed_dim, self.embed_dim),
        )
        self.ln_error = nn.LayerNorm(self.alphabet_size)
        self.t_embedder = DiTTimestepEmbedder(self.embed_dim)
        self.backbone = SharedTransformerBackbone(
            hidden_size=self.embed_dim,
            n_layers=self.n_transformer_layers,
            n_heads=self.n_heads,
            ffn_hidden_size=4 * self.embed_dim,
            dropout=self.attention_dropout,
            attention_mode="bidirectional",
            use_timestep_conditioning=True,
        )
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.prob_proj = nn.Linear(self.embed_dim, self.alphabet_size)

        _init_embedding_normal(self.x_values)

    def transformer_blocks(self):
        return self.backbone.transformer_blocks()

    def _compute_error(self, y: torch.Tensor, snr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        snr_t = torch.clamp(snr, min=1e-12).to(dtype=y.dtype, device=y.device)
        alphabet, d = self.x_values.weight.shape
        x_values = self.x_values.weight.to(dtype=y.dtype, device=y.device).view(1, 1, alphabet, d)
        error = 0.5 * d * torch.log(snr_t) - 0.5 * snr_t * ((y.unsqueeze(2) - x_values) ** 2).sum(dim=-1)
        return error, snr_t

    def forward_outputs(
        self,
        y: torch.Tensor,
        snr: torch.Tensor,
        *,
        t_embed: torch.Tensor | None = None,
        channel_tilt: torch.Tensor | None = None,
    ) -> TransformerModelOutput:
        if y.dim() != 3:
            raise ValueError(f"Expected y to be [B, S, D], got {tuple(y.shape)}")
        if snr.dim() != 3 or snr.shape[:2] != y.shape[:2] or snr.shape[2] != 1:
            raise ValueError(f"Expected snr to be [B, S, 1], got {tuple(snr.shape)}")
        _, seq_len, d_in = y.shape
        if d_in != self.embed_dim:
            raise ValueError(f"Expected input dim {self.embed_dim}, got {d_in}")
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")

        error, snr_t = self._compute_error(y, snr)
        cond = self.t_embedder(torch.log(snr_t).squeeze(-1)) if t_embed is None else t_embed
        hidden_input = self.error_proj(self.ln_error(error)) + cond
        hidden = self.backbone(hidden_input, t_embed=cond)

        emb_hat = self.out_proj(hidden)
        logits_ext = self.prob_proj(hidden)
        tilt = error if channel_tilt is None else channel_tilt
        return TransformerModelOutput(hidden=hidden, logits_ext=logits_ext, logits=apply_channel_tilt(logits_ext, tilt), emb_hat=emb_hat)

    def forward(
        self,
        y: torch.Tensor,
        snr: torch.Tensor,
        *,
        t_embed: torch.Tensor | None = None,
        channel_tilt: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.forward_outputs(y, snr, t_embed=t_embed, channel_tilt=channel_tilt)
        return outputs.emb_hat, outputs.logits_ext, outputs.logits


class CausalCrossEntropySequenceModel(nn.Module):
    """Causal token model aligned to the shared GPT-2-style backbone."""

    def __init__(
        self,
        *,
        alphabet_size: int,
        embed_dim: int,
        n_transformer_layers: int,
        n_heads: int,
        attention_dropout: float,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.alphabet_size = int(alphabet_size)
        self.embed_dim = int(embed_dim)
        self.n_transformer_layers = max(1, int(n_transformer_layers))
        self.n_heads = int(n_heads)
        self.max_seq_len = int(max_seq_len)
        self.x_values = nn.Embedding(self.alphabet_size, self.embed_dim)
        self.error_proj = nn.Linear(self.alphabet_size, self.embed_dim)
        self.ln_error = nn.LayerNorm(self.alphabet_size)
        self.backbone = SharedTransformerBackbone(
            hidden_size=self.embed_dim,
            n_layers=self.n_transformer_layers,
            n_heads=self.n_heads,
            ffn_hidden_size=4 * self.embed_dim,
            dropout=float(attention_dropout),
            attention_mode="causal",
        )
        self.prob_proj = nn.Linear(self.embed_dim, self.alphabet_size)
        _init_embedding_normal(self.x_values)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 2:
            raise ValueError(f"Expected tokens to be [B, S], got {tuple(tokens.shape)}")
        _, seq_len = tokens.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")
        token_emb = self.x_values(tokens.long())
        token_one_hot = F.one_hot(tokens.long(), num_classes=self.alphabet_size).to(dtype=token_emb.dtype)
        hidden = self.backbone(token_emb + self.error_proj(self.ln_error(token_one_hot)))
        return self.prob_proj(hidden)
