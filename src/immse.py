"""I-MMSE curve evaluation and area integration."""

from __future__ import annotations

from contextlib import contextmanager
import torch

from .bcjr import true_markov_mmse_bcjr_from_batch
from .channels import awgn
from .models import posterior_mean_from_logits
from .sources_iid import log_priors, sample_iid_states
from .sources_markov import MarkovChainSpec, sample_markov_sequences_torch


def integrate_half_mmse_curve(snr_grid: torch.Tensor | object, mmse_curve: torch.Tensor | object) -> float:
    snr_grid = torch.as_tensor(snr_grid, dtype=torch.float64)
    mmse_curve = torch.as_tensor(mmse_curve, dtype=torch.float64)
    if snr_grid.ndim != 1 or mmse_curve.ndim != 1 or snr_grid.shape[0] != mmse_curve.shape[0]:
        raise ValueError("snr_grid and mmse_curve must be 1D arrays with the same length")
    if snr_grid.shape[0] < 2:
        return 0.0
    if bool(torch.any(snr_grid <= 0)):
        raise ValueError("snr_grid must be strictly positive")

    order = torch.argsort(snr_grid)
    snr_sorted = snr_grid[order]
    mmse_sorted = mmse_curve[order]
    return float(torch.trapz(0.5 * mmse_sorted * snr_sorted, torch.log(snr_sorted)).item())


def relative_error(estimate: float, reference: float) -> float:
    return abs(float(estimate) - float(reference)) / max(abs(float(reference)), 1e-12)


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


@torch.no_grad()
def iid_true_mmse_at_snr(
    model: torch.nn.Module,
    snr: float,
    source_probs: torch.Tensor,
    device: torch.device | str,
    n_samples: int = 40_000,
    batch_size: int = 4096,
    seed: int | None = None,
) -> float:
    with _torch_seed_scope(seed):
        model.eval()
        total_se = 0.0
        total_n = 0
        remaining = n_samples
        x_values = model.x_values.weight.to(device=device)
        log_prior = log_priors(source_probs, device=device, dtype=x_values.dtype).unsqueeze(0)

        while remaining > 0:
            bs = min(batch_size, remaining)
            snr_t = torch.full((bs, 1), float(snr), device=device)
            states = sample_iid_states(bs, source_probs, device=device)
            clean = x_values[states]
            noisy = awgn(clean, snr_t)

            error = ((noisy.unsqueeze(1) - x_values.unsqueeze(0)) ** 2).sum(dim=-1)
            logits = log_prior + 0.5 * clean.shape[1] * torch.log(snr_t) - 0.5 * snr_t * error
            mean = posterior_mean_from_logits(logits, x_values)
            total_se += torch.sum((clean - mean) ** 2).item()
            total_n += bs
            remaining -= bs

        return total_se / max(total_n, 1)


@torch.no_grad()
def iid_model_mmse_at_snr(
    model: torch.nn.Module,
    snr: float,
    source_probs: torch.Tensor,
    device: torch.device | str,
    n_samples: int = 40_000,
    batch_size: int = 4096,
    seed: int | None = None,
    topk: int | None = None,
) -> float:
    with _torch_seed_scope(seed):
        model.eval()
        total_se = 0.0
        total_n = 0
        remaining = n_samples
        x_values = model.x_values.weight.to(device=device)

        while remaining > 0:
            bs = min(batch_size, remaining)
            snr_t = torch.full((bs, 1), float(snr), device=device)
            states = sample_iid_states(bs, source_probs, device=device)
            clean = x_values[states]
            noisy = awgn(clean, snr_t)
            _, _, logits = model(noisy, snr_t)
            mean = posterior_mean_from_logits(logits, x_values, topk=topk)
            total_se += torch.sum((clean - mean) ** 2).item()
            total_n += bs
            remaining -= bs

        return total_se / max(total_n, 1)


def iid_mmse_curves(
    model: torch.nn.Module,
    snr_grid: torch.Tensor,
    source_probs: torch.Tensor,
    device: torch.device | str,
    n_samples: int,
    batch_size: int,
    seed: int | None = None,
) -> dict[str, torch.Tensor]:
    base_seed = None if seed is None else int(seed)
    true_curve = torch.tensor(
        [
            iid_true_mmse_at_snr(
                model,
                snr,
                source_probs,
                device,
                n_samples=n_samples,
                batch_size=batch_size,
                seed=None if base_seed is None else base_seed + idx,
            )
            for idx, snr in enumerate(snr_grid)
        ],
        dtype=torch.float64,
    )
    model_curve = torch.tensor(
        [
            iid_model_mmse_at_snr(
                model,
                snr,
                source_probs,
                device,
                n_samples=n_samples,
                batch_size=batch_size,
                seed=None if base_seed is None else base_seed + 10_000 + idx,
            )
            for idx, snr in enumerate(snr_grid)
        ],
        dtype=torch.float64,
    )
    return {"true_mmse": true_curve, "model_mmse": model_curve}


def iid_model_mmse_curve(
    model: torch.nn.Module,
    snr_grid: torch.Tensor,
    source_probs: torch.Tensor,
    device: torch.device | str,
    n_samples: int,
    batch_size: int,
    seed: int | None = None,
    topk: int | None = None,
) -> torch.Tensor:
    base_seed = None if seed is None else int(seed)
    return torch.tensor(
        [
            iid_model_mmse_at_snr(
                model,
                snr,
                source_probs,
                device,
                n_samples=n_samples,
                batch_size=batch_size,
                seed=None if base_seed is None else base_seed + 10_000 + idx,
                topk=topk,
            )
            for idx, snr in enumerate(snr_grid)
        ],
        dtype=torch.float64,
    )


@torch.no_grad()
def iid_regression_mse_at_snr(
    model: torch.nn.Module,
    snr: float,
    source_probs: torch.Tensor,
    device: torch.device | str,
    n_samples: int = 40_000,
    batch_size: int = 4096,
    seed: int | None = None,
) -> float:
    with _torch_seed_scope(seed):
        model.eval()
        total_se = 0.0
        total_n = 0
        remaining = n_samples
        x_values = model.x_values.weight.to(device=device)

        while remaining > 0:
            bs = min(batch_size, remaining)
            snr_t = torch.full((bs, 1), float(snr), device=device)
            states = sample_iid_states(bs, source_probs, device=device)
            clean = x_values[states]
            noisy = awgn(clean, snr_t)
            emb_hat, _, _ = model(noisy, snr_t)
            total_se += torch.sum((clean - emb_hat) ** 2).item()
            total_n += bs
            remaining -= bs

        return total_se / max(total_n, 1)


def iid_regression_mse_curve(
    model: torch.nn.Module,
    snr_grid: torch.Tensor,
    source_probs: torch.Tensor,
    device: torch.device | str,
    n_samples: int,
    batch_size: int,
    seed: int | None = None,
) -> torch.Tensor:
    base_seed = None if seed is None else int(seed)
    return torch.tensor(
        [
            iid_regression_mse_at_snr(
                model,
                snr,
                source_probs,
                device,
                n_samples=n_samples,
                batch_size=batch_size,
                seed=None if base_seed is None else base_seed + 20_000 + idx,
            )
            for idx, snr in enumerate(snr_grid)
        ],
        dtype=torch.float64,
    )


@torch.no_grad()
def model_markov_mmse_at_snr(
    model: torch.nn.Module,
    snr: float,
    chain: MarkovChainSpec,
    device: torch.device | str,
    batch_size: int = 24,
    seq_len: int = 64,
    seed: int = 0,
    topk: int | None = None,
    sequence_permutation: torch.Tensor | None = None,
) -> float:
    model.eval()
    init_probs_t = chain.init_probs.to(dtype=torch.float32)
    transition_matrix_t = chain.transition_matrix.to(dtype=torch.float32)
    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(int(seed))
    uniforms = torch.rand((batch_size, seq_len), generator=cpu_generator, dtype=torch.float32)
    states = sample_markov_sequences_torch(
        num_sequences=batch_size,
        seq_len=seq_len,
        init_probs=init_probs_t,
        transition_matrix=transition_matrix_t,
        uniforms=uniforms,
    )
    if sequence_permutation is not None:
        permutation = torch.as_tensor(sequence_permutation, dtype=torch.long, device=states.device)
        if permutation.numel() != seq_len:
            raise ValueError(f"Expected sequence_permutation length {seq_len}, got {permutation.numel()}")
        states = states[:, permutation]
    constellation_t = model.x_values.weight.detach().to(dtype=torch.float32, device=device)
    clean_t = constellation_t[states.to(device=device)]
    snr_t = torch.full((batch_size, seq_len, 1), float(snr), dtype=clean_t.dtype, device=device)
    noisy_t = awgn(clean_t, snr_t)

    _, _, logits = model(noisy_t, snr_t)
    constellation_t = model.x_values.weight.to(dtype=noisy_t.dtype, device=noisy_t.device)
    mean = posterior_mean_from_logits(logits, constellation_t, topk=topk)
    mse = torch.mean(torch.sum((clean_t - mean) ** 2, dim=-1))
    return float(mse.item())


def markov_true_mmse_at_snr(
    model: torch.nn.Module,
    snr: float,
    chain: MarkovChainSpec,
    batch_size: int = 24,
    seq_len: int = 64,
    seed: int = 0,
) -> float:
    init_probs_t = chain.init_probs.to(dtype=torch.float32)
    transition_matrix_t = chain.transition_matrix.to(dtype=torch.float32)
    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(int(seed))
    uniforms = torch.rand((batch_size, seq_len), generator=cpu_generator, dtype=torch.float32)
    states = sample_markov_sequences_torch(
        num_sequences=batch_size,
        seq_len=seq_len,
        init_probs=init_probs_t,
        transition_matrix=transition_matrix_t,
        uniforms=uniforms,
    )
    constellation_t = model.x_values.weight.detach().cpu().to(dtype=torch.float32)
    clean = constellation_t[states]
    snr_t = torch.full((batch_size, seq_len, 1), float(snr), dtype=clean.dtype)
    noisy = awgn(clean, snr_t)
    return true_markov_mmse_bcjr_from_batch(
        snr=snr,
        clean=clean,
        noisy=noisy,
        init_probs=init_probs_t,
        transition_matrix=transition_matrix_t,
        constellation=constellation_t,
    )


def markov_mmse_curves(
    model: torch.nn.Module,
    snr_grid: torch.Tensor,
    chain: MarkovChainSpec,
    device: torch.device | str,
    batch_size: int,
    seq_len: int,
    seed: int,
    sequence_permutation: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    true_curve = []
    model_curve = []
    for idx, snr in enumerate(snr_grid):
        eval_seed = int(seed) + idx
        true_curve.append(markov_true_mmse_at_snr(model, float(snr), chain, batch_size=batch_size, seq_len=seq_len, seed=eval_seed))
        model_curve.append(
            model_markov_mmse_at_snr(
                model,
                float(snr),
                chain,
                device=device,
                batch_size=batch_size,
                seq_len=seq_len,
                seed=eval_seed,
                sequence_permutation=sequence_permutation,
            )
        )
    return {
        "true_mmse": torch.tensor(true_curve, dtype=torch.float64),
        "model_mmse": torch.tensor(model_curve, dtype=torch.float64),
    }


def markov_model_mmse_curve(
    model: torch.nn.Module,
    snr_grid: torch.Tensor,
    chain: MarkovChainSpec,
    device: torch.device | str,
    batch_size: int,
    seq_len: int,
    seed: int,
    topk: int | None = None,
    sequence_permutation: torch.Tensor | None = None,
) -> torch.Tensor:
    model_curve = []
    for idx, snr in enumerate(snr_grid):
        eval_seed = int(seed) + idx
        model_curve.append(
            model_markov_mmse_at_snr(
                model,
                float(snr),
                chain,
                device=device,
                batch_size=batch_size,
                seq_len=seq_len,
                seed=eval_seed,
                topk=topk,
                sequence_permutation=sequence_permutation,
            )
        )
    return torch.tensor(model_curve, dtype=torch.float64)


@torch.no_grad()
def markov_regression_mse_at_snr(
    model: torch.nn.Module,
    snr: float,
    chain: MarkovChainSpec,
    device: torch.device | str,
    batch_size: int = 24,
    seq_len: int = 64,
    seed: int = 0,
    sequence_permutation: torch.Tensor | None = None,
) -> float:
    model.eval()
    init_probs_t = chain.init_probs.to(dtype=torch.float32)
    transition_matrix_t = chain.transition_matrix.to(dtype=torch.float32)
    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(int(seed))
    uniforms = torch.rand((batch_size, seq_len), generator=cpu_generator, dtype=torch.float32)
    states = sample_markov_sequences_torch(
        num_sequences=batch_size,
        seq_len=seq_len,
        init_probs=init_probs_t,
        transition_matrix=transition_matrix_t,
        uniforms=uniforms,
    )
    if sequence_permutation is not None:
        permutation = torch.as_tensor(sequence_permutation, dtype=torch.long, device=states.device)
        if permutation.numel() != seq_len:
            raise ValueError(f"Expected sequence_permutation length {seq_len}, got {permutation.numel()}")
        states = states[:, permutation]
    constellation_t = model.x_values.weight.detach().to(dtype=torch.float32, device=device)
    clean_t = constellation_t[states.to(device=device)]
    snr_t = torch.full((batch_size, seq_len, 1), float(snr), dtype=clean_t.dtype, device=device)
    noisy_t = awgn(clean_t, snr_t)

    emb_hat, _, _ = model(noisy_t, snr_t)
    mse = torch.mean(torch.sum((clean_t - emb_hat) ** 2, dim=-1))
    return float(mse.item())


def markov_regression_mse_curve(
    model: torch.nn.Module,
    snr_grid: torch.Tensor,
    chain: MarkovChainSpec,
    device: torch.device | str,
    batch_size: int,
    seq_len: int,
    seed: int,
    sequence_permutation: torch.Tensor | None = None,
) -> torch.Tensor:
    model_curve = []
    for idx, snr in enumerate(snr_grid):
        eval_seed = int(seed) + idx
        model_curve.append(
            markov_regression_mse_at_snr(
                model,
                float(snr),
                chain,
                device=device,
                batch_size=batch_size,
                seq_len=seq_len,
                seed=eval_seed,
                sequence_permutation=sequence_permutation,
            )
        )
    return torch.tensor(model_curve, dtype=torch.float64)
