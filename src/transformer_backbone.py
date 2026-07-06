from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(slots=True)
class TransformerModelOutput:
    hidden: torch.Tensor
    logits_ext: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    emb_hat: torch.Tensor | None = None


def apply_channel_tilt(logits_ext: torch.Tensor, channel_tilt: torch.Tensor | None = None) -> torch.Tensor:
    if channel_tilt is None:
        return logits_ext
    tilt = torch.as_tensor(channel_tilt, dtype=logits_ext.dtype, device=logits_ext.device)
    if tilt.shape != logits_ext.shape:
        raise ValueError(f"Expected channel_tilt shape {tuple(logits_ext.shape)}, got {tuple(tilt.shape)}")
    return logits_ext + tilt


def dit_timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    timesteps_t = torch.as_tensor(timesteps, dtype=torch.float32)
    original_shape = timesteps_t.shape
    flat = timesteps_t.reshape(-1)
    half = dim // 2
    if half == 0:
        return torch.zeros((*original_shape, dim), dtype=torch.float32, device=timesteps_t.device)
    freqs = torch.exp(-math.log(float(max_period)) * torch.arange(half, device=flat.device, dtype=torch.float32) / half)
    args = flat.unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 != 0:
        emb = F.pad(emb, (0, 1))
    return emb.reshape(*original_shape, dim)


class DiTTimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.frequency_embedding_size = int(frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_embedding_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        emb = dit_timestep_embedding(timesteps, dim=self.frequency_embedding_size).to(dtype=self.mlp[0].weight.dtype)
        return self.mlp(emb)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)


def _apply_rope(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _, _, seq_len, head_dim = q.shape
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires even head_dim, got {head_dim}")
    half_dim = head_dim // 2
    positions = torch.arange(seq_len, device=q.device, dtype=torch.float32)
    inv_freq = 1.0 / (10_000 ** (torch.arange(half_dim, device=q.device, dtype=torch.float32) / half_dim))
    freqs = torch.einsum("i,j->ij", positions, inv_freq)
    angles = torch.repeat_interleave(freqs, repeats=2, dim=-1)
    cos = torch.cos(angles).to(dtype=q.dtype).view(1, 1, seq_len, head_dim)
    sin = torch.sin(angles).to(dtype=q.dtype).view(1, 1, seq_len, head_dim)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale) + shift


class GPT2StyleBlock(nn.Module):
    def __init__(self, *, hidden_size: int, n_heads: int, ffn_hidden_size: int, dropout: float, cond_dim: int | None) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.n_heads = int(n_heads)
        self.head_dim = self.hidden_size // self.n_heads
        if self.hidden_size % self.n_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by n_heads={n_heads}")
        if self.head_dim % 2 != 0:
            raise ValueError(f"RoPE requires hidden_size/n_heads to be even, got head_dim={self.head_dim}")

        self.norm_attn = nn.LayerNorm(self.hidden_size)
        self.qkv = nn.Linear(self.hidden_size, 3 * self.hidden_size)
        self.attn_out = nn.Linear(self.hidden_size, self.hidden_size)
        self.attn_dropout = nn.Dropout(float(dropout))
        self.resid_dropout = nn.Dropout(float(dropout))
        self.norm_mlp = nn.LayerNorm(self.hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(self.hidden_size, int(ffn_hidden_size)),
            nn.GELU(),
            nn.Linear(int(ffn_hidden_size), self.hidden_size),
        )
        self.ff_dropout = nn.Dropout(float(dropout))
        self.cond_proj = nn.Linear(int(cond_dim), 6 * self.hidden_size) if cond_dim is not None else None
        if self.cond_proj is not None:
            nn.init.zeros_(self.cond_proj.weight)
            nn.init.zeros_(self.cond_proj.bias)

    def _cond_chunks(
        self,
        cond: torch.Tensor | None,
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if cond is None:
            return None
        if self.cond_proj is None:
            raise ValueError("Received conditioning input for a block without conditioning support.")
        cond_t = torch.as_tensor(cond, dtype=dtype, device=device)
        if cond_t.ndim == 2:
            cond_proj = self.cond_proj(F.silu(cond_t)).unsqueeze(1)
        elif cond_t.ndim == 3:
            if cond_t.shape[:2] != (batch_size, seq_len):
                raise ValueError(f"Expected conditioning shape {(batch_size, seq_len, cond_t.shape[-1])}, got {tuple(cond_t.shape)}")
            cond_proj = self.cond_proj(F.silu(cond_t))
        else:
            raise ValueError(f"Expected conditioning to be [B, D] or [B, S, D], got {tuple(cond_t.shape)}")
        return cond_proj.chunk(6, dim=-1)

    @staticmethod
    def _masked_softmax(attn_scores: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return torch.softmax(attn_scores, dim=-1)
        masked_scores = attn_scores.masked_fill(mask, float("-inf"))
        attn_probs = torch.softmax(masked_scores, dim=-1)
        attn_probs = attn_probs.masked_fill(mask, 0.0)
        normalizer = attn_probs.sum(dim=-1, keepdim=True)
        return torch.where(normalizer > 0, attn_probs / normalizer.clamp_min(torch.finfo(attn_probs.dtype).tiny), torch.zeros_like(attn_probs))

    def _self_attention(self, x: torch.Tensor, *, is_causal: bool) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x).view(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        q, k = _apply_rope(q, k)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        attn_mask = None
        if is_causal:
            attn_mask = torch.triu(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool), diagonal=1)
        else:
            attn_mask = torch.eye(seq_len, device=x.device, dtype=torch.bool)
        attn_probs = self._masked_softmax(attn_scores, None if attn_mask is None else attn_mask.view(1, 1, seq_len, seq_len))
        attn_probs = self.attn_dropout(attn_probs)
        attn = torch.matmul(attn_probs, v)
        attn = attn.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.hidden_size)
        return self.resid_dropout(self.attn_out(attn))

    def forward(self, x: torch.Tensor, *, is_causal: bool, cond: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        chunks = self._cond_chunks(cond, batch_size=batch_size, seq_len=seq_len, device=x.device, dtype=x.dtype)

        attn_input = self.norm_attn(x)
        if chunks is None:
            attn_out = self._self_attention(attn_input, is_causal=is_causal)
            x = x + attn_out
            mlp_input = self.norm_mlp(x)
            mlp_out = self.ff_dropout(self.ff(mlp_input))
            return x + mlp_out

        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = chunks
        attn_out = self._self_attention(_modulate(attn_input, shift_attn, scale_attn), is_causal=is_causal)
        x = x + gate_attn * attn_out
        mlp_input = _modulate(self.norm_mlp(x), shift_mlp, scale_mlp)
        mlp_out = self.ff_dropout(self.ff(mlp_input))
        return x + gate_mlp * mlp_out


class SharedTransformerBackbone(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        n_layers: int,
        n_heads: int,
        ffn_hidden_size: int,
        dropout: float,
        attention_mode: str,
        use_timestep_conditioning: bool = False,
    ) -> None:
        super().__init__()
        if attention_mode not in {"causal", "bidirectional"}:
            raise ValueError(f"Unsupported attention_mode={attention_mode!r}")
        self.hidden_size = int(hidden_size)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.ffn_hidden_size = int(ffn_hidden_size)
        self.dropout = float(dropout)
        self.attention_mode = str(attention_mode)
        self.use_timestep_conditioning = bool(use_timestep_conditioning)
        cond_dim = self.hidden_size if self.use_timestep_conditioning else None
        self.input_dropout = nn.Dropout(self.dropout)
        self.blocks = nn.ModuleList(
            [
                GPT2StyleBlock(
                    hidden_size=self.hidden_size,
                    n_heads=self.n_heads,
                    ffn_hidden_size=self.ffn_hidden_size,
                    dropout=self.dropout,
                    cond_dim=cond_dim,
                )
                for _ in range(self.n_layers)
            ]
        )
        self.final_ln = nn.LayerNorm(self.hidden_size)

    def transformer_blocks(self) -> list[GPT2StyleBlock]:
        return list(self.blocks)

    def forward(self, hidden_states: torch.Tensor, *, t_embed: torch.Tensor | None = None) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(f"Expected hidden_states [B, S, H], got {tuple(hidden_states.shape)}")
        hidden = self.input_dropout(hidden_states)
        is_causal = self.attention_mode == "causal"
        for block in self.blocks:
            hidden = block(hidden, is_causal=is_causal, cond=t_embed)
        return self.final_ln(hidden)
