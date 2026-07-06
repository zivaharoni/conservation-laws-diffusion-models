"""Transformer models for language-model objective comparisons."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .models import _init_embedding_normal
from .transformer_backbone import (
    DiTTimestepEmbedder,
    SharedTransformerBackbone,
    TransformerModelOutput,
    apply_channel_tilt,
)


class _BaseTransformerLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        n_layers: int,
        n_heads: int,
        ffn_hidden_size: int,
        max_seq_len: int,
        dropout: float,
        attention_mode: str,
        use_timestep_conditioning: bool = False,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.ffn_hidden_size = int(ffn_hidden_size)
        self.max_seq_len = int(max_seq_len)
        self.dropout = float(dropout)
        self.attention_mode = str(attention_mode)
        self.backbone = SharedTransformerBackbone(
            hidden_size=self.hidden_size,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            ffn_hidden_size=self.ffn_hidden_size,
            dropout=self.dropout,
            attention_mode=self.attention_mode,
            use_timestep_conditioning=bool(use_timestep_conditioning),
        )

    def _validate_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 2:
            raise ValueError(f"Expected tokens to be [B, S], got {tuple(tokens.shape)}")
        if tokens.shape[1] > self.max_seq_len:
            raise ValueError(f"seq_len={tokens.shape[1]} exceeds max_seq_len={self.max_seq_len}")
        return tokens.long()

    def _resolve_timestep_condition(
        self,
        *,
        timesteps: torch.Tensor | None,
        t_embed: torch.Tensor | None,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        cond = t_embed
        if cond is None and timesteps is not None:
            if not hasattr(self, "t_embedder"):
                raise ValueError("Received timesteps for a model without timestep conditioning enabled.")
            timesteps_t = torch.as_tensor(timesteps, dtype=torch.float32, device=device)
            if timesteps_t.ndim == 3:
                if timesteps_t.shape != (batch_size, seq_len, 1):
                    raise ValueError(f"Expected timesteps shape {(batch_size, seq_len, 1)}, got {tuple(timesteps_t.shape)}")
                timesteps_t = timesteps_t.squeeze(-1)
            elif timesteps_t.ndim == 2:
                if timesteps_t.shape == (batch_size, 1):
                    timesteps_t = timesteps_t.squeeze(-1)
                elif timesteps_t.shape != (batch_size, seq_len):
                    raise ValueError(f"Expected timesteps shape {(batch_size, seq_len)} or {(batch_size, 1)}, got {tuple(timesteps_t.shape)}")
            elif timesteps_t.ndim != 1 or timesteps_t.shape[0] != batch_size:
                raise ValueError(f"Expected timesteps shape {(batch_size,)}, got {tuple(timesteps_t.shape)}")
            cond = self.t_embedder(timesteps_t)
        if cond is None:
            return None
        cond_t = torch.as_tensor(cond, dtype=dtype, device=device)
        if cond_t.ndim == 2 and cond_t.shape != (batch_size, self.hidden_size):
            raise ValueError(f"Expected t_embed shape {(batch_size, self.hidden_size)}, got {tuple(cond_t.shape)}")
        if cond_t.ndim == 3 and cond_t.shape != (batch_size, seq_len, self.hidden_size):
            raise ValueError(f"Expected t_embed shape {(batch_size, seq_len, self.hidden_size)}, got {tuple(cond_t.shape)}")
        if cond_t.ndim not in {2, 3}:
            raise ValueError(f"Expected t_embed rank 2 or 3, got {tuple(cond_t.shape)}")
        return cond_t

    @staticmethod
    def _add_timestep_condition(hidden_states: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        if cond is None:
            return hidden_states
        if cond.ndim == 2:
            return hidden_states + cond.unsqueeze(1)
        return hidden_states + cond


class CausalTransformerLM(_BaseTransformerLM):
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        n_layers: int,
        n_heads: int,
        ffn_hidden_size: int,
        max_seq_len: int,
        dropout: float = 0.0,
        use_timestep_conditioning: bool = False,
    ) -> None:
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            n_heads=n_heads,
            ffn_hidden_size=ffn_hidden_size,
            max_seq_len=max_seq_len,
            dropout=dropout,
            attention_mode="causal",
            use_timestep_conditioning=use_timestep_conditioning,
        )
        if bool(use_timestep_conditioning):
            self.t_embedder = DiTTimestepEmbedder(self.hidden_size)
        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.logits_bias = nn.Parameter(torch.zeros(self.vocab_size))
        _init_embedding_normal(self.token_embedding)

    def forward_outputs(
        self,
        tokens: torch.Tensor,
        *,
        timesteps: torch.Tensor | None = None,
        t_embed: torch.Tensor | None = None,
        channel_tilt: torch.Tensor | None = None,
    ) -> TransformerModelOutput:
        tokens_t = self._validate_tokens(tokens)
        token_hidden = self.token_embedding(tokens_t)
        cond = self._resolve_timestep_condition(
            timesteps=timesteps,
            t_embed=t_embed,
            batch_size=int(tokens_t.shape[0]),
            seq_len=int(tokens_t.shape[1]),
            device=tokens_t.device,
            dtype=token_hidden.dtype,
        )
        hidden = self.backbone(self._add_timestep_condition(token_hidden, cond), t_embed=cond)
        logits_ext = F.linear(hidden, self.token_embedding.weight, self.logits_bias)
        return TransformerModelOutput(hidden=hidden, logits_ext=logits_ext, logits=apply_channel_tilt(logits_ext, channel_tilt))

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        timesteps: torch.Tensor | None = None,
        t_embed: torch.Tensor | None = None,
        channel_tilt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_outputs(tokens, timesteps=timesteps, t_embed=t_embed, channel_tilt=channel_tilt).logits


class MaskedTransformerLM(_BaseTransformerLM):
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        n_layers: int,
        n_heads: int,
        ffn_hidden_size: int,
        max_seq_len: int,
        dropout: float = 0.0,
        use_timestep_conditioning: bool = False,
    ) -> None:
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            n_heads=n_heads,
            ffn_hidden_size=ffn_hidden_size,
            max_seq_len=max_seq_len,
            dropout=dropout,
            attention_mode="bidirectional",
            use_timestep_conditioning=use_timestep_conditioning,
        )
        if bool(use_timestep_conditioning):
            self.t_embedder = DiTTimestepEmbedder(self.hidden_size)
        self.input_vocab_size = self.vocab_size + 1
        self.mask_token_id = self.vocab_size
        self.token_embedding = nn.Embedding(self.input_vocab_size, self.hidden_size)
        self.logits_bias = nn.Parameter(torch.zeros(self.vocab_size))
        _init_embedding_normal(self.token_embedding)

    def forward_outputs(
        self,
        tokens: torch.Tensor,
        *,
        timesteps: torch.Tensor | None = None,
        t_embed: torch.Tensor | None = None,
        channel_tilt: torch.Tensor | None = None,
    ) -> TransformerModelOutput:
        tokens_t = self._validate_tokens(tokens)
        token_hidden = self.token_embedding(tokens_t)
        cond = self._resolve_timestep_condition(
            timesteps=timesteps,
            t_embed=t_embed,
            batch_size=int(tokens_t.shape[0]),
            seq_len=int(tokens_t.shape[1]),
            device=tokens_t.device,
            dtype=token_hidden.dtype,
        )
        hidden = self.backbone(self._add_timestep_condition(token_hidden, cond), t_embed=cond)
        logits_ext = F.linear(hidden, self.token_embedding.weight[: self.vocab_size], self.logits_bias)
        return TransformerModelOutput(hidden=hidden, logits_ext=logits_ext, logits=apply_channel_tilt(logits_ext, channel_tilt))

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        timesteps: torch.Tensor | None = None,
        t_embed: torch.Tensor | None = None,
        channel_tilt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_outputs(tokens, timesteps=timesteps, t_embed=t_embed, channel_tilt=channel_tilt).logits


class ContinuousDiffusionTransformerLM(_BaseTransformerLM):
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        n_layers: int,
        n_heads: int,
        ffn_hidden_size: int,
        max_seq_len: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            n_heads=n_heads,
            ffn_hidden_size=ffn_hidden_size,
            max_seq_len=max_seq_len,
            dropout=dropout,
            attention_mode="bidirectional",
            use_timestep_conditioning=True,
        )
        self.x_values = nn.Embedding(self.vocab_size, self.hidden_size)
        self.t_embedder = DiTTimestepEmbedder(self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.logits_bias = nn.Parameter(torch.zeros(self.vocab_size))
        _init_embedding_normal(self.x_values)

    def _gaussian_channel_tilt(self, noisy_embeddings: torch.Tensor, snr: torch.Tensor) -> torch.Tensor:
        snr_t = torch.clamp(snr, min=1.0e-12).to(dtype=noisy_embeddings.dtype, device=noisy_embeddings.device)
        x_values = self.x_values.weight.to(dtype=noisy_embeddings.dtype, device=noisy_embeddings.device)
        distances = torch.sum(
            (noisy_embeddings.unsqueeze(2) - x_values.view(1, 1, self.vocab_size, self.hidden_size)) ** 2,
            dim=-1,
        )
        return 0.5 * self.hidden_size * torch.log(snr_t) - 0.5 * snr_t * distances

    def forward_outputs(
        self,
        noisy_embeddings: torch.Tensor,
        snr: torch.Tensor,
        *,
        t_embed: torch.Tensor | None = None,
        channel_tilt: torch.Tensor | None = None,
    ) -> TransformerModelOutput:
        if noisy_embeddings.dim() != 3:
            raise ValueError(f"Expected noisy_embeddings to be [B, S, D], got {tuple(noisy_embeddings.shape)}")
        if snr.dim() != 3 or snr.shape[:2] != noisy_embeddings.shape[:2] or snr.shape[2] != 1:
            raise ValueError(f"Expected snr to be [B, S, 1], got {tuple(snr.shape)}")
        if noisy_embeddings.shape[2] != self.hidden_size:
            raise ValueError(f"Expected noisy embedding dim {self.hidden_size}, got {int(noisy_embeddings.shape[2])}.")
        if noisy_embeddings.shape[1] > self.max_seq_len:
            raise ValueError(f"seq_len={noisy_embeddings.shape[1]} exceeds max_seq_len={self.max_seq_len}")
        snr_t = torch.clamp(snr, min=1.0e-12).to(dtype=noisy_embeddings.dtype, device=noisy_embeddings.device)
        cond = self.t_embedder(torch.log(snr_t).squeeze(-1)) if t_embed is None else t_embed
        hidden = self.backbone(noisy_embeddings + cond, t_embed=cond)

        emb_hat = self.out_proj(hidden)
        logits_ext = F.linear(hidden, self.x_values.weight, self.logits_bias)
        tilt = self._gaussian_channel_tilt(noisy_embeddings, snr_t) if channel_tilt is None else channel_tilt
        return TransformerModelOutput(hidden=hidden, logits_ext=logits_ext, logits=apply_channel_tilt(logits_ext, tilt), emb_hat=emb_hat)

    def forward(
        self,
        noisy_embeddings: torch.Tensor,
        snr: torch.Tensor,
        *,
        t_embed: torch.Tensor | None = None,
        channel_tilt: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.forward_outputs(noisy_embeddings, snr, t_embed=t_embed, channel_tilt=channel_tilt)
        return outputs.emb_hat, outputs.logits_ext, outputs.logits
