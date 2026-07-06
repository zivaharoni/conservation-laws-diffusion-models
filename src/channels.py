"""AWGN and discrete channel helpers."""

from __future__ import annotations

import math
import torch


def clamp_snr(snr: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.clamp(snr, min=eps)


def awgn(x: torch.Tensor, snr: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    snr_safe = clamp_snr(snr, eps=eps)
    return x + torch.randn_like(x) / torch.sqrt(snr_safe)


def sample_log_uniform_snr(
    batch_shape: tuple[int, ...],
    snr_min: float,
    snr_max: float,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    log_snr_min = math.log(float(snr_min))
    log_snr_max = math.log(float(snr_max))
    if generator is None:
        uniforms = torch.rand(*batch_shape, device=device)
    else:
        uniforms = torch.rand(*batch_shape, generator=generator, dtype=torch.float32).to(device=device)
    return torch.exp(uniforms * (log_snr_max - log_snr_min) + log_snr_min)


# ---------------------------------------------------------------------------
# K-ary Symmetric Channel (BSC generalisation)
# ---------------------------------------------------------------------------
# Channel kernel W_t(y | x):
#   W_t(x | x) = t                    ("input is out w.p. t")
#   W_t(u | x) = (1-t)/(K-1)  u != x  ("w.p. 1-t uniform over the rest")
#
# The valid degradation range (uniform channel -> perfect channel) is t in
# [1/K, 1].  Training may use t in (0, 1) freely since NLL is well-defined,
# but the GEXIT conservation-law integral is only meaningful over [1/K, 1].
# ---------------------------------------------------------------------------


def sample_uniform_ksc_rates(
    batch_shape: tuple[int, ...],
    t_min: float,
    t_max: float,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample per-sequence pass-through rates t ~ Uniform[t_min, t_max]."""
    if generator is None:
        uniforms = torch.rand(*batch_shape, device=device)
    else:
        uniforms = torch.rand(*batch_shape, generator=generator, dtype=torch.float32).to(device=device)
    return uniforms * (t_max - t_min) + t_min


def apply_ksc_noise(
    tokens: torch.Tensor,
    t: torch.Tensor,
    alphabet_size: int,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply K-ary symmetric channel noise independently per token.

    Args:
        tokens:        [B, S] clean token indices in {0, ..., K-1}.
        t:             [B] pass-through probabilities (or scalar).
        alphabet_size: K, the number of symbols.
        generator:     optional CPU RNG for reproducibility.

    Returns:
        noisy_tokens:  [B, S] observed token indices (same dtype as tokens).
        correct_mask:  [B, S] bool, True where noisy == clean.
    """
    if tokens.dim() != 2:
        raise ValueError(f"Expected tokens [B, S], got {tuple(tokens.shape)}")
    batch_size, seq_len = tokens.shape
    K = int(alphabet_size)

    t_t = torch.as_tensor(t, dtype=torch.float32, device=tokens.device)
    if t_t.ndim == 0:
        t_t = t_t.expand(batch_size)
    if t_t.shape != (batch_size,):
        raise ValueError(f"Expected t shape ({batch_size},), got {tuple(t_t.shape)}")

    # Decide pass-through: shape [B, S]
    if generator is None:
        uniforms = torch.rand(batch_size, seq_len, dtype=torch.float32, device=tokens.device)
    else:
        uniforms = torch.rand(batch_size, seq_len, generator=generator, dtype=torch.float32).to(device=tokens.device)
    pass_through = uniforms < t_t.unsqueeze(1)  # [B, S]

    # For substituted positions: sample uniformly from K-1 other symbols.
    # Strategy: sample offset in {1, ..., K-1} then add modulo K.
    if generator is None:
        offsets = torch.randint(1, K, (batch_size, seq_len), device=tokens.device, dtype=tokens.dtype)
    else:
        offsets = torch.randint(1, K, (batch_size, seq_len), generator=generator, dtype=tokens.dtype).to(device=tokens.device)
    substitute_tokens = (tokens + offsets) % K

    noisy_tokens = torch.where(pass_through, tokens, substitute_tokens)
    correct_mask = noisy_tokens == tokens
    return noisy_tokens, correct_mask


def log_ksc_kernel(
    observed: torch.Tensor,
    t: torch.Tensor,
    alphabet_size: int,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Log channel kernel log W_t(y_i | x) for all hypotheses x.

    Given an observed symbol y_i, returns a [B, S, K] tensor where entry
    [b, i, x] = log W_t(observed[b,i] | x):
        log(t)             if x == observed[b, i]
        log((1-t)/(K-1))   otherwise

    Args:
        observed:      [B, S] long tensor of observed tokens in {0, ..., K-1}.
        t:             [B] pass-through probabilities (or scalar).
        alphabet_size: K.
        eps:           numerical floor for log arguments.

    Returns:
        [B, S, K] float tensor of log-kernel values.
    """
    if observed.dim() != 2:
        raise ValueError(f"Expected observed [B, S], got {tuple(observed.shape)}")
    batch_size, seq_len = observed.shape
    K = int(alphabet_size)

    t_t = torch.as_tensor(t, dtype=torch.float32, device=observed.device)
    if t_t.ndim == 0:
        t_t = t_t.expand(batch_size)

    t_safe = t_t.clamp(min=eps, max=1.0 - eps)  # [B]
    log_t = torch.log(t_safe)                                          # [B]
    log_noise = torch.log((1.0 - t_safe) / (K - 1))                   # [B]

    # [B, S, K] base = log_noise
    kernel = log_noise.view(batch_size, 1, 1).expand(batch_size, seq_len, K).clone()

    # Mask where x matches the observed symbol
    sym_range = torch.arange(K, device=observed.device).view(1, 1, K)  # [1, 1, K]
    match = sym_range == observed.unsqueeze(-1)                         # [B, S, K]

    kernel = torch.where(match, log_t.view(batch_size, 1, 1).expand_as(kernel), kernel)
    return kernel
