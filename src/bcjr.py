"""BCJR baseline utilities for Markov experiments."""

from __future__ import annotations

import torch


def _as_float64_tensor(value: torch.Tensor | object) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float64)


def bcjr_posteriors(
    y_seq: torch.Tensor,
    snr: float,
    init_probs: torch.Tensor,
    transition_matrix: torch.Tensor,
    constellation: torch.Tensor,
) -> torch.Tensor:
    y_seq = _as_float64_tensor(y_seq)
    init_probs = _as_float64_tensor(init_probs)
    transition_matrix = _as_float64_tensor(transition_matrix)
    constellation = _as_float64_tensor(constellation)

    t_len = y_seq.shape[0]
    log_pi = torch.log(torch.clamp(init_probs, min=1e-12))
    log_a = torch.log(torch.clamp(transition_matrix, min=1e-12))

    diff = y_seq[:, None, :] - constellation[None, :, :]
    log_b = -0.5 * float(snr) * torch.sum(diff * diff, dim=-1)

    alpha = torch.empty_like(log_b)
    alpha[0] = log_pi + log_b[0]
    for t in range(1, t_len):
        alpha[t] = log_b[t] + torch.logsumexp(alpha[t - 1][:, None] + log_a, dim=0)

    beta = torch.zeros_like(log_b)
    for t in range(t_len - 2, -1, -1):
        beta[t] = torch.logsumexp(log_a + log_b[t + 1][None, :] + beta[t + 1][None, :], dim=1)

    log_gamma = alpha + beta
    log_gamma = log_gamma - torch.logsumexp(log_gamma, dim=1, keepdim=True)
    post = torch.exp(log_gamma)
    post = torch.clamp(post, min=0.0)
    post = post / torch.clamp(post.sum(dim=1, keepdim=True), min=1e-15)
    return post


def true_markov_mmse_bcjr_from_batch(
    snr: float,
    clean: torch.Tensor,
    noisy: torch.Tensor,
    init_probs: torch.Tensor,
    transition_matrix: torch.Tensor,
    constellation: torch.Tensor,
) -> float:
    clean = _as_float64_tensor(clean)
    noisy = _as_float64_tensor(noisy)
    init_probs = _as_float64_tensor(init_probs)
    transition_matrix = _as_float64_tensor(transition_matrix)
    constellation = _as_float64_tensor(constellation)
    total_se = 0.0
    total_count = 0
    for n in range(clean.shape[0]):
        post = bcjr_posteriors(
            y_seq=noisy[n],
            snr=float(snr),
            init_probs=init_probs,
            transition_matrix=transition_matrix,
            constellation=constellation,
        )
        c_hat = post @ constellation
        se = torch.sum((clean[n] - c_hat) ** 2, dim=-1)
        total_se += float(torch.sum(se).item())
        total_count += se.numel()
    return total_se / max(total_count, 1)
