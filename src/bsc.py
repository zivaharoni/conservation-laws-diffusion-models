"""BSC-variant (K-ary Symmetric Channel) GEXIT utilities.

Channel model
-------------
W_t(y | x):
  t            if y == x   ("input is out w.p. t")
  (1-t)/(K-1)  if y != x   ("w.p. 1-t uniform over the rest")

For t in [1/K, 1] the channel degrades monotonically from the uniform
(uninformative) channel at t = 1/K to the perfect channel at t = 1.

GEXIT derivative (Theorem 1 / eq. 5 in the paper, specialised to K-SC)
-----------------------------------------------------------------------
The score function is  S_t(x, y) = d/dt log W_t(y|x):
  1/t        if y == x
  -1/(1-t)   if y != x

Plugging into the mismatched GEXIT formula gives

  d CE / dt = CE_correct(t) - CE_incorrect(t) + correction(t)

where
  CE_correct(t)   = E[-log P^θ(X|Y) | Y_i = X_i]   (CE at correct obs.)
  CE_incorrect(t) = E[-log P^θ(X|Y) | Y_i != X_i]  (CE at incorrect obs.)
  correction(t)   = E[P^θ(Y_i|Y)] / (t*(1-t)) - 1/(1-t)
  P^θ(Y_i|Y)     = model posterior prob. of the observed symbol Y_i.

The mismatched CE integral (eq. 6 in the paper) is then

  CE(P_X, P^theta_X) = -∫_{1/K}^{1}  d CE / dt  dt
                      = ∫_{1/K}^{1}  bsc_gexit(t)  dt

where  bsc_gexit(t) = CE_incorrect - CE_correct - correction.

For the K-SC, the theorem-1 / equation-(5) derivative decomposes as

  d CE / dt = score_nll_term(t) + mismatch_correction(t)

where
  score_nll_term(t) = CE_correct(t) - CE_incorrect(t)
  mismatch_correction(t) = correction(t).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from .channels import apply_ksc_noise, log_ksc_kernel, sample_uniform_ksc_rates
from .sources_markov import invert_permutation


def _model_outputs_with_tilt(model, observed: torch.Tensor, channel_tilt: torch.Tensor):
    if hasattr(model, "forward_outputs"):
        return model.forward_outputs(observed, channel_tilt=channel_tilt)
    logits_ext = model(observed)
    return logits_ext, logits_ext + channel_tilt


# ---------------------------------------------------------------------------
# Per-token NLL with log-channel-kernel reweighting
# ---------------------------------------------------------------------------


def per_token_nll_bsc(
    logits_ext: torch.Tensor,
    targets: torch.Tensor,
    observed: torch.Tensor,
    t: torch.Tensor,
    alphabet_size: int,
) -> torch.Tensor:
    """NLL under the K-SC posterior.

    Computes -log P^θ(X_i = x_i | Y) where

        P^θ(X_i | Y) ∝ P^θ(X_i | Y_{-i}) * W_t(Y_i | X_i)

    i.e. the model's extrinsic logits are reweighted by log W_t(y_i | x).

    Args:
        logits_ext:    [B, S, K] extrinsic logits from the model.
        targets:       [B, S]    true token indices (X).
        observed:      [B, S]    noisy observed tokens (Y).
        t:             [B] or scalar pass-through probabilities.
        alphabet_size: K.

    Returns:
        [B, S] per-token NLL (nats).
    """
    if logits_ext.dim() != 3:
        raise ValueError(f"Expected logits_ext [B, S, K], got {tuple(logits_ext.shape)}")
    if targets.shape != logits_ext.shape[:2]:
        raise ValueError(f"targets {tuple(targets.shape)} must match logits_ext batch/seq dims {tuple(logits_ext.shape[:2])}")

    kernel = log_ksc_kernel(observed, t, alphabet_size)          # [B, S, K]
    logits = logits_ext + kernel                                  # [B, S, K]

    B, S, K = logits.shape
    return F.cross_entropy(
        logits.reshape(-1, K),
        targets.reshape(-1),
        reduction="none",
    ).reshape(B, S)


# ---------------------------------------------------------------------------
# Metric accumulator (mirrors exit.py pattern)
# ---------------------------------------------------------------------------


def init_bsc_metric_accumulator() -> dict[str, float]:
    return {
        "total_loss_sum": 0.0,
        "total_token_count": 0.0,
        "correct_loss_sum": 0.0,
        "correct_token_count": 0.0,
        "incorrect_loss_sum": 0.0,
        "incorrect_token_count": 0.0,
        "model_prob_obs_sum": 0.0,    # sum of P^θ(Y_i | Y) over all tokens
        "sequence_count": 0.0,
        "t_sum": 0.0,
    }


def update_bsc_metric_accumulator(
    accumulator: dict[str, float],
    *,
    per_token_loss: torch.Tensor,
    correct_mask: torch.Tensor,
    logits_reweighted: torch.Tensor,
    observed: torch.Tensor,
    t_per_sequence: torch.Tensor,
) -> None:
    """Update accumulator with one batch.

    Args:
        per_token_loss:    [B, S] NLL values.
        correct_mask:      [B, S] bool, True where Y_i == X_i.
        logits_reweighted: [B, S, K] logits after adding log-channel-kernel.
        observed:          [B, S] observed token indices (Y).
        t_per_sequence:    [B] pass-through probabilities for this batch.
    """
    correct_f = correct_mask.to(dtype=per_token_loss.dtype)
    incorrect_f = (~correct_mask).to(dtype=per_token_loss.dtype)

    # P^θ(Y_i | Y) = softmax(logits_reweighted)[b, i, observed[b, i]]
    log_probs = F.log_softmax(logits_reweighted, dim=-1)                  # [B, S, K]
    obs_idx = observed.unsqueeze(-1)                                       # [B, S, 1]
    log_prob_obs = torch.gather(log_probs, dim=-1, index=obs_idx).squeeze(-1)  # [B, S]
    prob_obs = torch.exp(log_prob_obs)                                     # [B, S]

    accumulator["total_loss_sum"] += float(per_token_loss.sum().item())
    accumulator["total_token_count"] += float(per_token_loss.numel())
    accumulator["correct_loss_sum"] += float((per_token_loss * correct_f).sum().item())
    accumulator["correct_token_count"] += float(correct_f.sum().item())
    accumulator["incorrect_loss_sum"] += float((per_token_loss * incorrect_f).sum().item())
    accumulator["incorrect_token_count"] += float(incorrect_f.sum().item())
    accumulator["model_prob_obs_sum"] += float(prob_obs.sum().item())
    accumulator["sequence_count"] += float(per_token_loss.shape[0])
    accumulator["t_sum"] += float(t_per_sequence.sum().item())


def finalize_bsc_metric_accumulator(accumulator: dict[str, float], alphabet_size: int) -> dict[str, float]:
    """Compute scalar metrics from accumulated sums."""
    total = accumulator["total_token_count"]
    correct = accumulator["correct_token_count"]
    incorrect = accumulator["incorrect_token_count"]
    seq_count = accumulator["sequence_count"]

    mean_nll = float("nan") if total <= 0 else accumulator["total_loss_sum"] / total
    ce_correct = float("nan") if correct <= 0 else accumulator["correct_loss_sum"] / correct
    ce_incorrect = float("nan") if incorrect <= 0 else accumulator["incorrect_loss_sum"] / incorrect
    mean_prob_obs = float("nan") if total <= 0 else accumulator["model_prob_obs_sum"] / total
    mean_t = float("nan") if seq_count <= 0 else accumulator["t_sum"] / seq_count

    score_nll_term = ce_correct - ce_incorrect

    # Equation (5): d CE / dt = score_nll_term + mismatch_correction
    # with mismatch_correction = mean_prob_obs / (t*(1-t)) - 1/(1-t).
    # Equation (6) then integrates -d CE / dt, which we keep as bsc_gexit
    # for compatibility with the older curve-integration helpers.
    t = mean_t
    if not (0 < t < 1):
        mismatch_correction = float("nan")
        ce_derivative_nats = float("nan")
        gexit = float("nan")
    else:
        mismatch_correction = mean_prob_obs / (t * (1.0 - t)) - 1.0 / (1.0 - t)
        ce_derivative_nats = score_nll_term + mismatch_correction
        gexit = -ce_derivative_nats

    return {
        "loss_nats": mean_nll,
        "ce_correct_nats": ce_correct,
        "ce_incorrect_nats": ce_incorrect,
        "score_nll_term_nats": score_nll_term,
        "mismatch_correction_nats": mismatch_correction,
        "ce_derivative_nats": ce_derivative_nats,
        "bsc_gexit": gexit,
        "mean_prob_obs": mean_prob_obs,
        "mean_t": mean_t,
        "correct_tokens": int(correct),
        "incorrect_tokens": int(incorrect),
        "total_tokens": int(total),
        "empirical_correct_rate": float("nan") if total <= 0 else correct / total,
    }


# ---------------------------------------------------------------------------
# Model evaluation at a fixed t
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_model_bsc(
    model: torch.nn.Module,
    sample_tokens,
    *,
    batch_size: int,
    seq_len: int,
    eval_batches: int,
    alphabet_size: int,
    device: torch.device | str,
    fixed_t: float,
    seed: int | None = None,
    steps_per_batch: int = 1,
) -> dict[str, float]:
    """Evaluate model under the K-SC at a fixed pass-through rate.

    Args:
        model:         BidirectionalMaskedTokenModel (or compatible) that
                       accepts noisy token indices and returns [B, S, K] logits.
        sample_tokens: callable (batch_size, seq_len, device) -> [B, S] long.
        fixed_t:       pass-through probability for this evaluation run.
        seed:          optional RNG seed for reproducibility.

    Returns:
        dict with the equation-(5) terms, the specialized BSC derivative,
        and the corresponding `bsc_gexit = - d CE / dt`.
    """
    model.eval()
    accumulator = init_bsc_metric_accumulator()
    K = int(alphabet_size)
    steps_per_batch = int(steps_per_batch)
    if steps_per_batch <= 0:
        raise ValueError(f"steps_per_batch must be positive, got {steps_per_batch}.")

    cpu_gen = None
    if seed is not None:
        cpu_gen = torch.Generator(device="cpu")
        cpu_gen.manual_seed(int(seed))

    for _ in range(int(eval_batches)):
        tokens = sample_tokens(batch_size, seq_len, device).long()
        t_batch = torch.full((batch_size,), float(fixed_t), dtype=torch.float32, device=device)

        for _ in range(steps_per_batch):
            noisy_tokens, correct_mask = apply_ksc_noise(tokens, t_batch, K, generator=cpu_gen)

            kernel = log_ksc_kernel(noisy_tokens, t_batch, K)            # [B, S, K]
            outputs = _model_outputs_with_tilt(model, noisy_tokens, kernel)
            if isinstance(outputs, tuple):
                logits_ext, logits_reweighted = outputs
            else:
                logits_ext, logits_reweighted = outputs.logits_ext, outputs.logits
            del logits_ext

            losses = F.cross_entropy(
                logits_reweighted.reshape(-1, K),
                tokens.reshape(-1),
                reduction="none",
            ).reshape(batch_size, seq_len)

            update_bsc_metric_accumulator(
                accumulator,
                per_token_loss=losses.detach(),
                correct_mask=correct_mask,
                logits_reweighted=logits_reweighted.detach(),
                observed=noisy_tokens,
                t_per_sequence=t_batch,
            )

    return finalize_bsc_metric_accumulator(accumulator, K)


# ---------------------------------------------------------------------------
# GEXIT curve over a t grid
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_model_bsc_curve(
    model: torch.nn.Module,
    sample_tokens,
    *,
    t_grid: torch.Tensor,
    batch_size: int,
    seq_len: int,
    eval_batches: int,
    alphabet_size: int,
    device: torch.device | str,
    seed: int | None = None,
    steps_per_batch: int = 1,
) -> dict[str, torch.Tensor]:
    """Evaluate the BSC GEXIT curve over a grid of t values.

    Returns a dict with keys:
      ce_correct, ce_incorrect, score_nll_term, mismatch_correction,
      ce_derivative, bsc_gexit, mean_prob_obs
    each a 1-D tensor aligned with t_grid.
    """
    base_seed = None if seed is None else int(seed)
    ce_correct_list = []
    ce_incorrect_list = []
    score_nll_term_list = []
    mismatch_correction_list = []
    ce_derivative_list = []
    gexit_list = []
    prob_obs_list = []

    for idx, t_val in enumerate(torch.as_tensor(t_grid, dtype=torch.float64).tolist()):
        metrics = evaluate_model_bsc(
            model,
            sample_tokens,
            batch_size=batch_size,
            seq_len=seq_len,
            eval_batches=eval_batches,
            alphabet_size=alphabet_size,
            device=device,
            fixed_t=float(t_val),
            seed=None if base_seed is None else base_seed + idx,
            steps_per_batch=steps_per_batch,
        )
        ce_correct_list.append(metrics["ce_correct_nats"])
        ce_incorrect_list.append(metrics["ce_incorrect_nats"])
        score_nll_term_list.append(metrics["score_nll_term_nats"])
        mismatch_correction_list.append(metrics["mismatch_correction_nats"])
        ce_derivative_list.append(metrics["ce_derivative_nats"])
        gexit_list.append(metrics["bsc_gexit"])
        prob_obs_list.append(metrics["mean_prob_obs"])

    return {
        "ce_correct": torch.tensor(ce_correct_list, dtype=torch.float64),
        "ce_incorrect": torch.tensor(ce_incorrect_list, dtype=torch.float64),
        "score_nll_term": torch.tensor(score_nll_term_list, dtype=torch.float64),
        "mismatch_correction": torch.tensor(mismatch_correction_list, dtype=torch.float64),
        "ce_derivative": torch.tensor(ce_derivative_list, dtype=torch.float64),
        "bsc_gexit": torch.tensor(gexit_list, dtype=torch.float64),
        "mean_prob_obs": torch.tensor(prob_obs_list, dtype=torch.float64),
    }


# ---------------------------------------------------------------------------
# Integration: mismatched CE integral (eq. 6)
# ---------------------------------------------------------------------------


def integrate_bsc_gexit_curve(
    t_grid: torch.Tensor | object,
    bsc_gexit_curve: torch.Tensor | object,
) -> float:
    """Integrate the BSC GEXIT curve to estimate CE(P_X, P^theta_X).

    Computes  ∫_{t_min}^{t_max}  bsc_gexit(t)  dt  via trapezoidal rule,
    where t_grid spans [1/K, 1] (the proper degradation range).

    NaN entries (e.g. from t values very close to 1 where almost no incorrect
    tokens appear) are excluded from the integration.

    Args:
        t_grid:          1-D tensor of t values.
        bsc_gexit_curve: 1-D tensor of bsc_gexit values aligned with t_grid.

    Returns:
        Scalar area estimate (nats).
    """
    t_g = torch.as_tensor(t_grid, dtype=torch.float64)
    g = torch.as_tensor(bsc_gexit_curve, dtype=torch.float64)
    if t_g.ndim != 1 or g.ndim != 1 or t_g.shape[0] != g.shape[0]:
        raise ValueError("t_grid and bsc_gexit_curve must be 1-D arrays of equal length")
    if t_g.shape[0] < 2:
        return 0.0
    order = torch.argsort(t_g)
    t_sorted = t_g[order]
    g_sorted = g[order]
    # Drop NaN entries (can occur near t=1 when too few incorrect tokens exist)
    valid = ~torch.isnan(g_sorted)
    if valid.sum() < 2:
        return float("nan")
    return float(torch.trapz(g_sorted[valid], t_sorted[valid]).item())


def save_bsc_curve_plot(
    output_path,
    *,
    t_grid: torch.Tensor,
    true_bsc_gexit: torch.Tensor,
    model_bsc_gexit: torch.Tensor,
    true_area: float,
    model_area: float,
    entropy: float,
    title: str,
) -> None:
    t_grid = torch.as_tensor(t_grid, dtype=torch.float64).cpu()
    true_bsc_gexit = torch.as_tensor(true_bsc_gexit, dtype=torch.float64).cpu()
    model_bsc_gexit = torch.as_tensor(model_bsc_gexit, dtype=torch.float64).cpu()

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(
        t_grid,
        true_bsc_gexit,
        linewidth=2.4,
        color="tab:blue",
        label=f"Analytic reference area={true_area:.4f} entropy={entropy:.4f}",
    )
    ax.plot(
        t_grid,
        model_bsc_gexit,
        linestyle="-.",
        linewidth=2.0,
        color="tab:orange",
        label=f"Model area={model_area:.4f}",
    )
    ax.fill_between(t_grid, torch.zeros_like(t_grid), true_bsc_gexit, color="tab:blue", alpha=0.22)
    ax.fill_between(t_grid, torch.zeros_like(t_grid), model_bsc_gexit, color="tab:orange", alpha=0.18)
    ax.set_xlim(float(t_grid.min().item()), float(t_grid.max().item()))
    ax.set_xlabel("Pass-through rate t")
    ax.set_ylabel("BSC GEXIT")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reference curve from the true IID posterior
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_reference_iid_bsc_curve(
    source_probs: torch.Tensor,
    *,
    t_grid: torch.Tensor,
    alphabet_size: int,
    batch_size: int,
    seq_len: int,
    eval_batches: int,
    seed: int | None = None,
) -> dict[str, torch.Tensor]:
    """Compute the TRUE BSC GEXIT curve for an IID source.

    For an IID source with known marginals, the extrinsic posterior equals
    the prior P(X_i), so the true channel posterior is just

        P(X_i | Y_i = y) ∝ P(X_i) * W_t(y | X_i).

    We sample from the source and channel, then compute the analytically
    exact posterior NLL at each position.

    Returns a dict with ce_correct, ce_incorrect, score_nll_term,
    mismatch_correction, ce_derivative, bsc_gexit aligned with t_grid.
    """
    from .sources_iid import entropy_nats

    probs_t = torch.as_tensor(source_probs, dtype=torch.float64, device="cpu")
    probs_t = probs_t / probs_t.sum()
    K = int(alphabet_size)

    base_seed = None if seed is None else int(seed)
    ce_correct_list = []
    ce_incorrect_list = []
    score_nll_term_list = []
    mismatch_correction_list = []
    ce_derivative_list = []
    gexit_list = []

    gen = None
    if base_seed is not None:
        gen = torch.Generator(device="cpu")

    for idx, t_val in enumerate(torch.as_tensor(t_grid, dtype=torch.float64).tolist()):
        t = float(t_val)
        if gen is not None:
            gen.manual_seed(base_seed + idx)

        total_ce_correct = 0.0
        total_ce_incorrect = 0.0
        total_prob_obs = 0.0
        n_correct = 0
        n_incorrect = 0
        n_total = 0

        for _ in range(int(eval_batches)):
            # Sample tokens iid
            states = torch.multinomial(
                probs_t.float(),
                num_samples=batch_size * seq_len,
                replacement=True,
                generator=gen,
            ).view(batch_size, seq_len)

            t_batch = torch.full((batch_size,), t, dtype=torch.float32)
            noisy, correct_mask = apply_ksc_noise(states, t_batch, K, generator=gen)

            # True extrinsic posterior = source prior (IID, no context)
            # log P(X_i | Y_i = y) ∝ log P(X_i) + log W_t(y | X_i)
            log_prior = torch.log(probs_t.clamp(min=1e-12)).float()   # [K]
            kernel = log_ksc_kernel(noisy, t_batch, K)                 # [B, S, K]
            log_posterior = log_prior.view(1, 1, K) + kernel           # [B, S, K]

            # NLL = cross-entropy against true labels
            B, S, _K = log_posterior.shape
            nll = F.cross_entropy(
                log_posterior.reshape(-1, K),
                states.reshape(-1),
                reduction="none",
            ).reshape(B, S)

            # P^true(Y_i | Y) = posterior probability of observed symbol
            log_probs_norm = F.log_softmax(log_posterior, dim=-1)
            log_prob_obs = torch.gather(log_probs_norm, -1, noisy.unsqueeze(-1)).squeeze(-1)
            prob_obs = torch.exp(log_prob_obs)

            correct_f = correct_mask.float()
            incorrect_f = (~correct_mask).float()
            n_corr = int(correct_f.sum().item())
            n_incorr = int(incorrect_f.sum().item())

            total_ce_correct += float((nll * correct_f).sum().item())
            total_ce_incorrect += float((nll * incorrect_f).sum().item())
            total_prob_obs += float(prob_obs.sum().item())
            n_correct += n_corr
            n_incorrect += n_incorr
            n_total += batch_size * seq_len

        ce_c = float("nan") if n_correct == 0 else total_ce_correct / n_correct
        ce_i = float("nan") if n_incorrect == 0 else total_ce_incorrect / n_incorrect
        mean_prob_obs = float("nan") if n_total == 0 else total_prob_obs / n_total
        score_nll_term = ce_c - ce_i

        if not (0 < t < 1):
            mismatch_correction = float("nan")
            ce_derivative = float("nan")
            gexit = float("nan")
        else:
            mismatch_correction = mean_prob_obs / (t * (1.0 - t)) - 1.0 / (1.0 - t)
            ce_derivative = score_nll_term + mismatch_correction
            gexit = -ce_derivative

        ce_correct_list.append(ce_c)
        ce_incorrect_list.append(ce_i)
        score_nll_term_list.append(score_nll_term)
        mismatch_correction_list.append(mismatch_correction)
        ce_derivative_list.append(ce_derivative)
        gexit_list.append(gexit)

    return {
        "ce_correct": torch.tensor(ce_correct_list, dtype=torch.float64),
        "ce_incorrect": torch.tensor(ce_incorrect_list, dtype=torch.float64),
        "score_nll_term": torch.tensor(score_nll_term_list, dtype=torch.float64),
        "mismatch_correction": torch.tensor(mismatch_correction_list, dtype=torch.float64),
        "ce_derivative": torch.tensor(ce_derivative_list, dtype=torch.float64),
        "bsc_gexit": torch.tensor(gexit_list, dtype=torch.float64),
    }


def markov_ksc_posteriors(
    observed_tokens: torch.Tensor,
    t_value: float,
    *,
    init_probs: torch.Tensor,
    transition_matrix: torch.Tensor,
    alphabet_size: int,
) -> torch.Tensor:
    observed_t = torch.as_tensor(observed_tokens, dtype=torch.long, device="cpu")
    if observed_t.dim() != 1:
        raise ValueError(f"Expected observed_tokens [S], got {tuple(observed_t.shape)}")

    init_probs_t = torch.as_tensor(init_probs, dtype=torch.float64, device="cpu")
    transition_t = torch.as_tensor(transition_matrix, dtype=torch.float64, device="cpu")
    K = int(alphabet_size)
    seq_len = int(observed_t.shape[0])

    t_safe = float(max(min(float(t_value), 1.0 - 1e-12), 1e-12))
    log_pi = torch.log(torch.clamp(init_probs_t, min=1e-12))
    log_a = torch.log(torch.clamp(transition_t, min=1e-12))

    log_b = log_ksc_kernel(
        observed_t.view(1, seq_len),
        torch.tensor([t_safe], dtype=torch.float32),
        K,
    ).squeeze(0).to(dtype=torch.float64, device="cpu")

    alpha = torch.empty_like(log_b)
    alpha[0] = log_pi + log_b[0]
    for pos in range(1, seq_len):
        alpha[pos] = log_b[pos] + torch.logsumexp(alpha[pos - 1][:, None] + log_a, dim=0)

    beta = torch.zeros_like(log_b)
    for pos in range(seq_len - 2, -1, -1):
        beta[pos] = torch.logsumexp(log_a + log_b[pos + 1][None, :] + beta[pos + 1][None, :], dim=1)

    log_gamma = alpha + beta
    log_gamma = log_gamma - torch.logsumexp(log_gamma, dim=1, keepdim=True)
    posteriors = torch.exp(log_gamma)
    return posteriors / torch.clamp(posteriors.sum(dim=1, keepdim=True), min=1e-15)


def markov_ksc_posteriors_batch(
    observed_tokens: torch.Tensor,
    t_value: float,
    *,
    init_probs: torch.Tensor,
    transition_matrix: torch.Tensor,
    alphabet_size: int,
    sequence_permutation: torch.Tensor | None = None,
) -> torch.Tensor:
    observed_t = torch.as_tensor(observed_tokens, dtype=torch.long, device="cpu")
    if observed_t.dim() != 2:
        raise ValueError(f"Expected observed_tokens [B, S], got {tuple(observed_t.shape)}")

    permutation_t = None if sequence_permutation is None else torch.as_tensor(sequence_permutation, dtype=torch.long, device="cpu")
    inverse_permutation_t = None if permutation_t is None else invert_permutation(permutation_t).to(device="cpu")

    posteriors = []
    for batch_idx in range(observed_t.shape[0]):
        observed_seq = observed_t[batch_idx]
        if inverse_permutation_t is not None:
            observed_seq = observed_seq[inverse_permutation_t]
        posterior_chain = markov_ksc_posteriors(
            observed_seq,
            float(t_value),
            init_probs=init_probs,
            transition_matrix=transition_matrix,
            alphabet_size=alphabet_size,
        )
        if permutation_t is not None:
            posterior_chain = posterior_chain[permutation_t]
        posteriors.append(posterior_chain)
    return torch.stack(posteriors, dim=0)


@torch.no_grad()
def evaluate_reference_markov_bsc_curve(
    sample_tokens,
    *,
    init_probs: torch.Tensor,
    transition_matrix: torch.Tensor,
    t_grid: torch.Tensor,
    alphabet_size: int,
    batch_size: int,
    seq_len: int,
    eval_batches: int,
    seed: int | None = None,
    sequence_permutation: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    K = int(alphabet_size)
    base_seed = None if seed is None else int(seed)
    ce_correct_list = []
    ce_incorrect_list = []
    score_nll_term_list = []
    mismatch_correction_list = []
    ce_derivative_list = []
    gexit_list = []

    gen = None
    if base_seed is not None:
        gen = torch.Generator(device="cpu")

    for idx, t_val in enumerate(torch.as_tensor(t_grid, dtype=torch.float64).tolist()):
        t = float(t_val)
        if gen is not None:
            gen.manual_seed(base_seed + idx)

        total_ce_correct = 0.0
        total_ce_incorrect = 0.0
        total_prob_obs = 0.0
        n_correct = 0
        n_incorrect = 0
        n_total = 0

        for _ in range(int(eval_batches)):
            states = sample_tokens(batch_size, seq_len, "cpu").long().cpu()
            t_batch = torch.full((batch_size,), t, dtype=torch.float32)
            noisy, correct_mask = apply_ksc_noise(states, t_batch, K, generator=gen)

            posteriors = markov_ksc_posteriors_batch(
                noisy,
                t,
                init_probs=init_probs,
                transition_matrix=transition_matrix,
                alphabet_size=K,
                sequence_permutation=sequence_permutation,
            )

            target_probs = torch.gather(posteriors, dim=-1, index=states.unsqueeze(-1)).squeeze(-1)
            nll = -torch.log(torch.clamp(target_probs, min=1e-12))

            prob_obs = torch.gather(posteriors, dim=-1, index=noisy.unsqueeze(-1)).squeeze(-1)

            correct_f = correct_mask.to(dtype=torch.float64)
            incorrect_f = (~correct_mask).to(dtype=torch.float64)
            n_corr = int(correct_f.sum().item())
            n_incorr = int(incorrect_f.sum().item())

            total_ce_correct += float((nll * correct_f).sum().item())
            total_ce_incorrect += float((nll * incorrect_f).sum().item())
            total_prob_obs += float(prob_obs.sum().item())
            n_correct += n_corr
            n_incorrect += n_incorr
            n_total += batch_size * seq_len

        ce_c = float("nan") if n_correct == 0 else total_ce_correct / n_correct
        ce_i = float("nan") if n_incorrect == 0 else total_ce_incorrect / n_incorrect
        mean_prob_obs = float("nan") if n_total == 0 else total_prob_obs / n_total
        score_nll_term = ce_c - ce_i

        if not (0 < t < 1):
            mismatch_correction = float("nan")
            ce_derivative = float("nan")
            gexit = float("nan")
        else:
            mismatch_correction = mean_prob_obs / (t * (1.0 - t)) - 1.0 / (1.0 - t)
            ce_derivative = score_nll_term + mismatch_correction
            gexit = -ce_derivative

        ce_correct_list.append(ce_c)
        ce_incorrect_list.append(ce_i)
        score_nll_term_list.append(score_nll_term)
        mismatch_correction_list.append(mismatch_correction)
        ce_derivative_list.append(ce_derivative)
        gexit_list.append(gexit)

    return {
        "ce_correct": torch.tensor(ce_correct_list, dtype=torch.float64),
        "ce_incorrect": torch.tensor(ce_incorrect_list, dtype=torch.float64),
        "score_nll_term": torch.tensor(score_nll_term_list, dtype=torch.float64),
        "mismatch_correction": torch.tensor(mismatch_correction_list, dtype=torch.float64),
        "ce_derivative": torch.tensor(ce_derivative_list, dtype=torch.float64),
        "bsc_gexit": torch.tensor(gexit_list, dtype=torch.float64),
    }
