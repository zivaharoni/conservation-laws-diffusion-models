"""Masking-EXIT utilities, exact references, and plotting helpers."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "gexit-diffusion-matplotlib"))

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from .immse import relative_error
from .sources_iid import entropy_nats
from .sources_markov import invert_permutation


@contextmanager
def _torch_seed_scope(seed: int | None):
    if seed is None:
        yield
        return

    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(int(seed))
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def sample_uniform_mask_rates(
    batch_size: int,
    *,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if generator is None:
        return torch.rand((batch_size,), dtype=torch.float32, device=device)
    rates = torch.rand((batch_size,), generator=generator, dtype=torch.float32)
    return rates.to(device=device)


def apply_sequence_masking(
    tokens: torch.Tensor,
    s_per_sequence: torch.Tensor | float,
    *,
    mask_token_id: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if tokens.dim() != 2:
        raise ValueError(f"Expected tokens to be [B, S], got {tuple(tokens.shape)}")

    batch_size, seq_len = tokens.shape
    s_t = torch.as_tensor(s_per_sequence, dtype=torch.float32, device=tokens.device)
    if s_t.ndim == 0:
        s_t = torch.full((batch_size,), float(s_t.item()), dtype=torch.float32, device=tokens.device)
    if s_t.shape != (batch_size,):
        raise ValueError(f"Expected s_per_sequence shape {(batch_size,)}, got {tuple(s_t.shape)}")

    if generator is None:
        uniforms = torch.rand((batch_size, seq_len), dtype=torch.float32, device=tokens.device)
    elif torch.device(generator.device).type == tokens.device.type:
        uniforms = torch.rand((batch_size, seq_len), generator=generator, dtype=torch.float32, device=tokens.device)
    else:
        uniforms = torch.rand((batch_size, seq_len), generator=generator, dtype=torch.float32)
        uniforms = uniforms.to(device=tokens.device)
    mask = uniforms < s_t.unsqueeze(1)

    empty_rows = ~mask.any(dim=1)
    forced_count = int(empty_rows.sum().item())
    if forced_count > 0:
        row_idx = torch.nonzero(empty_rows, as_tuple=False).squeeze(1)
        if generator is None:
            col_idx = torch.randint(seq_len, (forced_count,), device=tokens.device)
        elif torch.device(generator.device).type == tokens.device.type:
            col_idx = torch.randint(seq_len, (forced_count,), generator=generator, dtype=torch.long, device=tokens.device)
        else:
            col_idx = torch.randint(seq_len, (forced_count,), generator=generator, dtype=torch.long)
            col_idx = col_idx.to(device=tokens.device)
        mask[row_idx, col_idx] = True

    masked_tokens = tokens.clone()
    masked_tokens[mask] = int(mask_token_id)
    return masked_tokens, mask, forced_count


def per_token_nll(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.dim() != 3:
        raise ValueError(f"Expected logits to be [B, S, K], got {tuple(logits.shape)}")
    if targets.shape != logits.shape[:2]:
        raise ValueError(
            f"Expected targets shape {tuple(logits.shape[:2])} to match logits batch/seq dims, got {tuple(targets.shape)}"
        )
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none").reshape(
        targets.shape
    )


def init_exit_metric_accumulator() -> dict[str, float]:
    return {
        "total_loss_sum": 0.0,
        "total_token_count": 0.0,
        "masked_loss_sum": 0.0,
        "masked_token_count": 0.0,
        "unmasked_loss_sum": 0.0,
        "unmasked_token_count": 0.0,
        "sequence_count": 0.0,
        "s_sum": 0.0,
        "forced_mask_sequences": 0.0,
    }


def update_exit_metric_accumulator(
    accumulator: dict[str, float],
    *,
    per_token_loss: torch.Tensor,
    mask: torch.Tensor,
    s_per_sequence: torch.Tensor,
    forced_mask_sequences: int,
) -> None:
    mask_f = mask.to(dtype=per_token_loss.dtype)
    unmasked_f = (~mask).to(dtype=per_token_loss.dtype)

    accumulator["total_loss_sum"] += float(per_token_loss.sum().item())
    accumulator["total_token_count"] += float(per_token_loss.numel())
    accumulator["masked_loss_sum"] += float((per_token_loss * mask_f).sum().item())
    accumulator["masked_token_count"] += float(mask_f.sum().item())
    accumulator["unmasked_loss_sum"] += float((per_token_loss * unmasked_f).sum().item())
    accumulator["unmasked_token_count"] += float(unmasked_f.sum().item())
    accumulator["sequence_count"] += float(mask.shape[0])
    accumulator["s_sum"] += float(s_per_sequence.sum().item())
    accumulator["forced_mask_sequences"] += float(forced_mask_sequences)


def finalize_exit_metric_accumulator(accumulator: dict[str, float]) -> dict[str, float]:
    masked_count = accumulator["masked_token_count"]
    unmasked_count = accumulator["unmasked_token_count"]
    total_count = accumulator["total_token_count"]
    sequence_count = accumulator["sequence_count"]
    masked_ce = float("nan") if masked_count <= 0 else accumulator["masked_loss_sum"] / masked_count
    unmasked_ce = float("nan") if unmasked_count <= 0 else accumulator["unmasked_loss_sum"] / unmasked_count
    masked_term = 0.0 if masked_count <= 0 else masked_ce
    unmasked_term = 0.0 if unmasked_count <= 0 else unmasked_ce
    return {
        "loss_nats": 0.0 if total_count <= 0 else accumulator["total_loss_sum"] / total_count,
        "masked_ce_nats": masked_ce,
        "unmasked_ce_nats": unmasked_ce,
        "exit_gap_nats": masked_term - unmasked_term,
        "mean_s": 0.0 if sequence_count <= 0 else accumulator["s_sum"] / sequence_count,
        "empirical_mask_rate": 0.0 if total_count <= 0 else masked_count / total_count,
        "forced_mask_sequences": int(accumulator["forced_mask_sequences"]),
        "forced_mask_fraction": 0.0
        if sequence_count <= 0
        else accumulator["forced_mask_sequences"] / sequence_count,
        "masked_tokens": int(masked_count),
        "unmasked_tokens": int(unmasked_count),
        "total_tokens": int(total_count),
    }


def integrate_exit_curve(s_grid: torch.Tensor | object, exit_curve: torch.Tensor | object) -> float:
    s_grid_t = torch.as_tensor(s_grid, dtype=torch.float64)
    exit_curve_t = torch.as_tensor(exit_curve, dtype=torch.float64)
    if s_grid_t.ndim != 1 or exit_curve_t.ndim != 1 or s_grid_t.shape[0] != exit_curve_t.shape[0]:
        raise ValueError("s_grid and exit_curve must be 1D arrays with the same length")
    if s_grid_t.shape[0] < 2:
        return 0.0
    if bool(torch.any(s_grid_t < 0.0)) or bool(torch.any(s_grid_t > 1.0)):
        raise ValueError("s_grid must lie in [0, 1]")

    order = torch.argsort(s_grid_t)
    return float(torch.trapz(exit_curve_t[order], s_grid_t[order]).item())


def save_exit_curve_plot(
    output_path: Path,
    *,
    s_grid: torch.Tensor,
    true_exit: torch.Tensor,
    model_exit: torch.Tensor,
    true_area: float,
    model_area: float,
    entropy: float,
    title: str,
) -> None:
    s_grid = torch.as_tensor(s_grid, dtype=torch.float64).cpu()
    true_exit = torch.as_tensor(true_exit, dtype=torch.float64).cpu()
    model_exit = torch.as_tensor(model_exit, dtype=torch.float64).cpu()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(
        s_grid,
        true_exit,
        linewidth=2.4,
        color="tab:blue",
        label=f"Reference area={true_area:.4f} entropy={entropy:.4f}",
    )
    ax.plot(
        s_grid,
        model_exit,
        linestyle="-.",
        linewidth=2.0,
        color="tab:orange",
        label=f"Learned model area={model_area:.4f}",
    )
    ax.fill_between(s_grid, torch.zeros_like(s_grid), true_exit, color="tab:blue", alpha=0.22)
    ax.fill_between(s_grid, torch.zeros_like(s_grid), model_exit, color="tab:orange", alpha=0.18)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Mask rate s")
    ax.set_ylabel("masked CE - unmasked CE")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def iid_exact_posteriors(observed_tokens: torch.Tensor, mask: torch.Tensor, source_probs: torch.Tensor) -> torch.Tensor:
    observed = torch.as_tensor(observed_tokens, dtype=torch.long)
    mask_t = torch.as_tensor(mask, dtype=torch.bool, device=observed.device)
    squeeze = False
    if observed.dim() == 1:
        observed = observed.unsqueeze(0)
        mask_t = mask_t.unsqueeze(0)
        squeeze = True
    if observed.shape != mask_t.shape:
        raise ValueError(f"Expected observed_tokens and mask to match, got {tuple(observed.shape)} and {tuple(mask_t.shape)}")

    probs = torch.as_tensor(source_probs, dtype=torch.float64, device=observed.device)
    probs = probs / torch.clamp(probs.sum(), min=1e-12)
    batch_size, seq_len = observed.shape
    posteriors = probs.view(1, 1, -1).expand(batch_size, seq_len, -1).clone()
    if bool((~mask_t).any()):
        observed_rows, observed_cols = torch.nonzero(~mask_t, as_tuple=True)
        posteriors[observed_rows, observed_cols] = 0.0
        posteriors[observed_rows, observed_cols, observed[observed_rows, observed_cols]] = 1.0
    return posteriors.squeeze(0) if squeeze else posteriors


def markov_masked_posteriors(
    observed_tokens: torch.Tensor,
    mask: torch.Tensor,
    init_probs: torch.Tensor,
    transition_matrix: torch.Tensor,
) -> torch.Tensor:
    observed_t = torch.as_tensor(observed_tokens, dtype=torch.long, device="cpu")
    mask_t = torch.as_tensor(mask, dtype=torch.bool, device="cpu")
    if observed_t.dim() != 1 or mask_t.dim() != 1 or observed_t.shape != mask_t.shape:
        raise ValueError("observed_tokens and mask must be matching 1D tensors")

    init_probs_t = torch.as_tensor(init_probs, dtype=torch.float64, device="cpu")
    transition_t = torch.as_tensor(transition_matrix, dtype=torch.float64, device="cpu")
    seq_len = int(observed_t.shape[0])
    n_states = int(init_probs_t.shape[0])

    log_pi = torch.log(torch.clamp(init_probs_t, min=1e-12))
    log_a = torch.log(torch.clamp(transition_t, min=1e-12))
    log_b = torch.zeros((seq_len, n_states), dtype=torch.float64)
    observed_rows = torch.nonzero(~mask_t, as_tuple=False).squeeze(1)
    if observed_rows.numel() > 0:
        log_b[observed_rows] = float("-inf")
        log_b[observed_rows, observed_t[observed_rows]] = 0.0

    alpha = torch.empty_like(log_b)
    alpha[0] = log_pi + log_b[0]
    for t in range(1, seq_len):
        alpha[t] = log_b[t] + torch.logsumexp(alpha[t - 1][:, None] + log_a, dim=0)

    beta = torch.zeros_like(log_b)
    for t in range(seq_len - 2, -1, -1):
        beta[t] = torch.logsumexp(log_a + log_b[t + 1][None, :] + beta[t + 1][None, :], dim=1)

    log_gamma = alpha + beta
    log_gamma = log_gamma - torch.logsumexp(log_gamma, dim=1, keepdim=True)
    posteriors = torch.exp(log_gamma)
    return posteriors / torch.clamp(posteriors.sum(dim=1, keepdim=True), min=1e-15)


def markov_masked_posteriors_batch(
    observed_tokens: torch.Tensor,
    mask: torch.Tensor,
    init_probs: torch.Tensor,
    transition_matrix: torch.Tensor,
    *,
    sequence_permutation: torch.Tensor | None = None,
) -> torch.Tensor:
    observed_t = torch.as_tensor(observed_tokens, dtype=torch.long, device="cpu")
    mask_t = torch.as_tensor(mask, dtype=torch.bool, device="cpu")
    if observed_t.dim() != 2 or mask_t.dim() != 2 or observed_t.shape != mask_t.shape:
        raise ValueError("observed_tokens and mask must be matching [B, S] tensors")

    permutation_t = None if sequence_permutation is None else torch.as_tensor(sequence_permutation, dtype=torch.long, device="cpu")
    inverse_permutation_t = None if permutation_t is None else invert_permutation(permutation_t).to(device="cpu")
    posteriors = []
    for batch_idx in range(observed_t.shape[0]):
        observed_seq = observed_t[batch_idx]
        mask_seq = mask_t[batch_idx]
        if inverse_permutation_t is not None:
            observed_seq = observed_seq[inverse_permutation_t]
            mask_seq = mask_seq[inverse_permutation_t]
        posterior_chain = markov_masked_posteriors(
            observed_tokens=observed_seq,
            mask=mask_seq,
            init_probs=init_probs,
            transition_matrix=transition_matrix,
        )
        if permutation_t is not None:
            posterior_chain = posterior_chain[permutation_t]
        posteriors.append(posterior_chain)
    return torch.stack(posteriors, dim=0)


def posterior_nll_from_true_states(posteriors: torch.Tensor, true_states: torch.Tensor) -> torch.Tensor:
    posteriors_t = torch.as_tensor(posteriors, dtype=torch.float64)
    true_states_t = torch.as_tensor(true_states, dtype=torch.long, device=posteriors_t.device)
    if posteriors_t.shape[:2] != true_states_t.shape:
        raise ValueError(
            f"Expected posteriors batch/seq dims {tuple(posteriors_t.shape[:2])} to match states {tuple(true_states_t.shape)}"
        )
    target_probs = torch.gather(posteriors_t, dim=-1, index=true_states_t.unsqueeze(-1)).squeeze(-1)
    return -torch.log(torch.clamp(target_probs, min=1e-12))


@torch.no_grad()
def evaluate_model_exit(
    model: torch.nn.Module,
    sample_tokens,
    *,
    batch_size: int,
    seq_len: int,
    eval_batches: int,
    device: torch.device | str,
    mask_token_id: int,
    seed: int | None = None,
    fixed_s: float | None = None,
    steps_per_batch: int = 1,
) -> dict[str, float]:
    model.eval()
    accumulator = init_exit_metric_accumulator()
    steps_per_batch = int(steps_per_batch)
    if steps_per_batch <= 0:
        raise ValueError(f"steps_per_batch must be positive, got {steps_per_batch}.")
    cpu_generator = None
    if seed is not None:
        cpu_generator = torch.Generator(device="cpu")
        cpu_generator.manual_seed(int(seed))

    for _ in range(int(eval_batches)):
        tokens = sample_tokens(batch_size, seq_len, device).long()
        for _ in range(steps_per_batch):
            if fixed_s is None:
                s_per_sequence = sample_uniform_mask_rates(batch_size, device=device, generator=cpu_generator)
            else:
                s_per_sequence = torch.full((batch_size,), float(fixed_s), dtype=torch.float32, device=device)
            masked_tokens, mask, forced_count = apply_sequence_masking(
                tokens,
                s_per_sequence,
                mask_token_id=mask_token_id,
                generator=cpu_generator,
            )
            logits = model(masked_tokens)
            losses = per_token_nll(logits, tokens)
            update_exit_metric_accumulator(
                accumulator,
                per_token_loss=losses.detach(),
                mask=mask,
                s_per_sequence=s_per_sequence.detach(),
                forced_mask_sequences=forced_count,
            )
    return finalize_exit_metric_accumulator(accumulator)


@torch.no_grad()
def evaluate_model_exit_curve(
    model: torch.nn.Module,
    sample_tokens,
    *,
    s_grid: torch.Tensor,
    batch_size: int,
    seq_len: int,
    eval_batches: int,
    device: torch.device | str,
    mask_token_id: int,
    seed: int | None = None,
    steps_per_batch: int = 1,
) -> dict[str, torch.Tensor]:
    base_seed = None if seed is None else int(seed)
    masked_curve = []
    unmasked_curve = []
    exit_curve = []
    for idx, s in enumerate(torch.as_tensor(s_grid, dtype=torch.float64).tolist()):
        metrics = evaluate_model_exit(
            model,
            sample_tokens,
            batch_size=batch_size,
            seq_len=seq_len,
            eval_batches=eval_batches,
            device=device,
            mask_token_id=mask_token_id,
            seed=None if base_seed is None else base_seed + idx,
            fixed_s=float(s),
            steps_per_batch=steps_per_batch,
        )
        masked_curve.append(metrics["masked_ce_nats"])
        unmasked_curve.append(metrics["unmasked_ce_nats"])
        exit_curve.append(metrics["exit_gap_nats"])
    return {
        "masked_ce": torch.tensor(masked_curve, dtype=torch.float64),
        "unmasked_ce": torch.tensor(unmasked_curve, dtype=torch.float64),
        "exit_gap": torch.tensor(exit_curve, dtype=torch.float64),
    }


@torch.no_grad()
def evaluate_reference_iid_exit_curve(
    source_probs: torch.Tensor,
    *,
    s_grid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    entropy = float(entropy_nats(source_probs))
    s_grid_t = torch.as_tensor(s_grid, dtype=torch.float64)
    masked_curve = torch.full_like(s_grid_t, entropy)
    unmasked_curve = torch.zeros_like(s_grid_t)
    return {
        "masked_ce": masked_curve,
        "unmasked_ce": unmasked_curve,
        "exit_gap": masked_curve - unmasked_curve,
    }


@torch.no_grad()
def evaluate_reference_markov_exit(
    sample_tokens,
    *,
    init_probs: torch.Tensor,
    transition_matrix: torch.Tensor,
    batch_size: int,
    seq_len: int,
    eval_batches: int,
    seed: int | None = None,
    fixed_s: float | None = None,
    sequence_permutation: torch.Tensor | None = None,
) -> dict[str, float]:
    accumulator = init_exit_metric_accumulator()
    cpu_generator = None
    if seed is not None:
        cpu_generator = torch.Generator(device="cpu")
        cpu_generator.manual_seed(int(seed))

    for _ in range(int(eval_batches)):
        tokens = sample_tokens(batch_size, seq_len, "cpu").long().cpu()
        if fixed_s is None:
            s_per_sequence = sample_uniform_mask_rates(batch_size, device="cpu", generator=cpu_generator)
        else:
            s_per_sequence = torch.full((batch_size,), float(fixed_s), dtype=torch.float32)
        masked_tokens, mask, forced_count = apply_sequence_masking(
            tokens,
            s_per_sequence,
            mask_token_id=-1,
            generator=cpu_generator,
        )
        posteriors = markov_masked_posteriors_batch(
            masked_tokens,
            mask,
            init_probs=init_probs,
            transition_matrix=transition_matrix,
            sequence_permutation=sequence_permutation,
        )
        losses = posterior_nll_from_true_states(posteriors, tokens)
        update_exit_metric_accumulator(
            accumulator,
            per_token_loss=losses,
            mask=mask,
            s_per_sequence=s_per_sequence,
            forced_mask_sequences=forced_count,
        )
    return finalize_exit_metric_accumulator(accumulator)


@torch.no_grad()
def evaluate_reference_markov_exit_curve(
    sample_tokens,
    *,
    init_probs: torch.Tensor,
    transition_matrix: torch.Tensor,
    s_grid: torch.Tensor,
    batch_size: int,
    seq_len: int,
    eval_batches: int,
    seed: int | None = None,
    sequence_permutation: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    base_seed = None if seed is None else int(seed)
    masked_curve = []
    unmasked_curve = []
    exit_curve = []
    for idx, s in enumerate(torch.as_tensor(s_grid, dtype=torch.float64).tolist()):
        metrics = evaluate_reference_markov_exit(
            sample_tokens,
            init_probs=init_probs,
            transition_matrix=transition_matrix,
            batch_size=batch_size,
            seq_len=seq_len,
            eval_batches=eval_batches,
            seed=None if base_seed is None else base_seed + idx,
            fixed_s=float(s),
            sequence_permutation=sequence_permutation,
        )
        masked_curve.append(metrics["masked_ce_nats"])
        unmasked_curve.append(metrics["unmasked_ce_nats"])
        exit_curve.append(metrics["exit_gap_nats"])
    return {
        "masked_ce": torch.tensor(masked_curve, dtype=torch.float64),
        "unmasked_ce": torch.tensor(unmasked_curve, dtype=torch.float64),
        "exit_gap": torch.tensor(exit_curve, dtype=torch.float64),
    }


def curve_area_metrics(
    *,
    s_grid: torch.Tensor,
    reference_exit: torch.Tensor,
    model_exit: torch.Tensor,
    reference_entropy: float,
) -> dict[str, float]:
    reference_area = integrate_exit_curve(s_grid, reference_exit)
    model_area = integrate_exit_curve(s_grid, model_exit)
    return {
        "reference_entropy_nats": float(reference_entropy),
        "reference_exit_area_nats": float(reference_area),
        "reference_abs_error": abs(float(reference_area) - float(reference_entropy)),
        "reference_rel_error": relative_error(reference_area, reference_entropy),
        "eval_exit_area_nats": float(model_area),
        "model_abs_error": abs(float(model_area) - float(reference_entropy)),
        "model_rel_error": relative_error(model_area, reference_entropy),
    }
