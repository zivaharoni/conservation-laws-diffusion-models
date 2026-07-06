"""First-order Markov source helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from .channels import sample_log_uniform_snr
from .data_common import EvaluationParams, TrainingBatch
from .sources_iid import entropy_nats


@dataclass(frozen=True)
class MarkovChainSpec:
    init_probs: torch.Tensor
    transition_matrix: torch.Tensor
    entropy_rate_nats: float
    stationary_dist: torch.Tensor


def sample_markov_init_probs(alphabet_size: int, dirichlet_alpha: float = 1.0, seed: int = 0) -> torch.Tensor:
    concentration = torch.full((alphabet_size,), float(dirichlet_alpha), dtype=torch.float64)
    rng_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(int(seed))
        gamma_init = torch.distributions.Gamma(concentration, torch.ones_like(concentration)).sample()
        init_probs = gamma_init / gamma_init.sum()
    finally:
        torch.random.set_rng_state(rng_state)
    return init_probs


def sample_markov_transition_matrix(alphabet_size: int, dirichlet_alpha: float = 1.0, seed: int = 0) -> torch.Tensor:
    concentration = torch.full((alphabet_size,), float(dirichlet_alpha), dtype=torch.float64)
    rng_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(int(seed))
        gamma_trans = torch.distributions.Gamma(
            concentration.expand(alphabet_size, -1),
            torch.ones((alphabet_size, alphabet_size), dtype=torch.float64),
        ).sample()
        transition_matrix = gamma_trans / gamma_trans.sum(dim=1, keepdim=True)
    finally:
        torch.random.set_rng_state(rng_state)
    return transition_matrix


def sample_sequence_permutation(seq_len: int, seed: int = 0) -> torch.Tensor:
    rng_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(int(seed))
        permutation = torch.randperm(int(seq_len), dtype=torch.long)
    finally:
        torch.random.set_rng_state(rng_state)
    return permutation


def invert_permutation(permutation: torch.Tensor) -> torch.Tensor:
    permutation_t = torch.as_tensor(permutation, dtype=torch.long)
    inverse = torch.empty_like(permutation_t)
    inverse[permutation_t] = torch.arange(permutation_t.numel(), dtype=torch.long, device=permutation_t.device)
    return inverse


def stationary_distribution(transition_matrix: torch.Tensor, tol: float = 1e-12, max_iter: int = 20_000) -> torch.Tensor:
    k = transition_matrix.shape[0]
    pi = torch.full((k,), 1.0 / k, dtype=transition_matrix.dtype, device=transition_matrix.device)
    for _ in range(max_iter):
        pi_next = pi @ transition_matrix
        if torch.max(torch.abs(pi_next - pi)).item() < tol:
            pi = pi_next
            break
        pi = pi_next
    pi = torch.clamp(pi, min=1e-15)
    pi = pi / pi.sum()
    return pi


def markov_entropy_rate_nats(transition_matrix: torch.Tensor) -> tuple[float, torch.Tensor]:
    pi = stationary_distribution(transition_matrix)
    row_entropy = -(transition_matrix * torch.log(torch.clamp(transition_matrix, min=1e-12))).sum(dim=1)
    entropy_rate = float((pi * row_entropy).sum().item())
    return entropy_rate, pi


def build_markov_chain(
    alphabet_size: int,
    dirichlet_alpha: float,
    seed: int,
    init_distribution: str = "stationary",
) -> MarkovChainSpec:
    transition_matrix = sample_markov_transition_matrix(
        alphabet_size=alphabet_size,
        dirichlet_alpha=dirichlet_alpha,
        seed=seed,
    )
    entropy_rate, stationary_dist = markov_entropy_rate_nats(transition_matrix)
    init_distribution = str(init_distribution).lower()
    if init_distribution == "stationary":
        init_probs = stationary_dist.clone()
    elif init_distribution == "sampled":
        init_probs = sample_markov_init_probs(
            alphabet_size=alphabet_size,
            dirichlet_alpha=dirichlet_alpha,
            seed=seed + 1,
        )
    else:
        raise ValueError(
            f"Unsupported init_distribution={init_distribution!r}; expected 'stationary' or 'sampled'."
        )
    return MarkovChainSpec(
        init_probs=init_probs,
        transition_matrix=transition_matrix,
        entropy_rate_nats=entropy_rate,
        stationary_dist=stationary_dist,
    )


def _sample_categorical_from_probs(probs: torch.Tensor, uniforms: torch.Tensor) -> torch.Tensor:
    cdf = probs.cumsum(dim=-1)
    cdf[..., -1] = 1.0
    values = uniforms.contiguous().unsqueeze(-1)
    return torch.searchsorted(cdf, values, right=False).squeeze(-1)


def sample_markov_sequences_torch(
    num_sequences: int,
    seq_len: int,
    init_probs: torch.Tensor,
    transition_matrix: torch.Tensor,
    uniforms: torch.Tensor,
) -> torch.Tensor:
    if uniforms.shape != (num_sequences, seq_len):
        raise ValueError(f"Expected uniforms shape {(num_sequences, seq_len)}, got {tuple(uniforms.shape)}")

    states = torch.empty((num_sequences, seq_len), dtype=torch.long, device=uniforms.device)
    init_row = init_probs.unsqueeze(0).expand(num_sequences, -1)
    states[:, 0] = _sample_categorical_from_probs(init_row, uniforms[:, 0])
    for t in range(1, seq_len):
        probs_t = transition_matrix[states[:, t - 1]]
        states[:, t] = _sample_categorical_from_probs(probs_t, uniforms[:, t])
    return states


@dataclass(slots=True)
class MarkovSourceData:
    alphabet_size: int
    dirichlet_alpha: float
    seed: int
    train_seq_len: int
    eval_seq_len: int
    init_distribution: str = "stationary"
    chain: MarkovChainSpec = field(init=False)
    title: str = field(init=False)
    reference_label: str = field(init=False)
    reference_unit: str = field(init=False)
    run_label: str = field(init=False)

    def __post_init__(self) -> None:
        self.init_distribution = str(self.init_distribution).lower()
        self.chain = build_markov_chain(
            alphabet_size=int(self.alphabet_size),
            dirichlet_alpha=float(self.dirichlet_alpha),
            seed=int(self.seed),
            init_distribution=self.init_distribution,
        )
        self.title = "Markov MMSE Curves"
        self.reference_label = "Entropy-rate ground truth"
        self.reference_unit = "nats/symbol"
        self.run_label = "markov"

    @property
    def reference_entropy_nats(self) -> float:
        return float(self.chain.entropy_rate_nats)

    @property
    def stationary_entropy_nats(self) -> float:
        return entropy_nats(self.chain.stationary_dist)

    def data_attributes(self) -> list[tuple[str, str]]:
        return [
            ("alphabet_size", str(int(self.alphabet_size))),
            ("dirichlet_alpha", f"{float(self.dirichlet_alpha):.4f}"),
            ("init_distribution", self.init_distribution),
            ("init_probs", _format_tensor_preview(self.chain.init_probs)),
            ("stationary_dist", _format_tensor_preview(self.chain.stationary_dist)),
            ("transition_row0", _format_tensor_preview(self.chain.transition_matrix[0])),
        ]

    def model_init_kwargs(self) -> dict[str, object]:
        return {
            "alphabet_size": int(self.alphabet_size),
            "max_seq_len": max(int(self.train_seq_len), int(self.eval_seq_len)),
        }

    def dummy_forward_inputs(
        self,
        model: torch.nn.Module,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = 2
        embed_dim = int(model.embed_dim)
        seq_len = min(4, int(getattr(model, "max_seq_len", max(int(self.train_seq_len), int(self.eval_seq_len)))))
        y = torch.zeros((batch_size, seq_len, embed_dim), device=device, dtype=dtype)
        snr = torch.ones((batch_size, seq_len, 1), device=device, dtype=dtype)
        return y, snr

    def make_training_batch_sampler(
        self,
        *,
        device: torch.device | str,
        batch_size: int,
        snr_min: float,
        snr_max: float,
        seed: int,
    ) -> Callable[[], TrainingBatch]:
        init_probs = torch.as_tensor(self.chain.init_probs, dtype=torch.float32, device=device)
        transition_matrix = torch.as_tensor(self.chain.transition_matrix, dtype=torch.float32, device=device)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        seq_len = int(self.train_seq_len)

        def _sample() -> TrainingBatch:
            snr = sample_log_uniform_snr((batch_size, 1, 1), snr_min, snr_max, device=device)
            snr = snr.expand(-1, seq_len, -1)
            uniforms = torch.rand((batch_size, seq_len), generator=generator, dtype=torch.float32).to(device=device)
            states = sample_markov_sequences_torch(
                num_sequences=batch_size,
                seq_len=seq_len,
                init_probs=init_probs,
                transition_matrix=transition_matrix,
                uniforms=uniforms,
            )
            return TrainingBatch(states=states, snr=snr)

        return _sample

    def reference_curves(
        self,
        *,
        model: torch.nn.Module,
        snr_grid: torch.Tensor,
        device: torch.device | str,
        params: EvaluationParams,
        seed: int,
    ) -> dict[str, torch.Tensor]:
        from .immse import markov_mmse_curves

        return markov_mmse_curves(
            model=model,
            snr_grid=snr_grid,
            chain=self.chain,
            device=device,
            batch_size=params.batch_size,
            seq_len=params.seq_len,
            seed=seed,
        )

    def model_curve(
        self,
        *,
        model: torch.nn.Module,
        snr_grid: torch.Tensor,
        device: torch.device | str,
        params: EvaluationParams,
        expectation_method: str,
        expectation_topk: int | None,
        seed: int,
    ) -> torch.Tensor:
        from .immse import markov_model_mmse_curve, markov_regression_mse_curve

        if expectation_method == "probs":
            return markov_model_mmse_curve(
                model=model,
                snr_grid=snr_grid,
                chain=self.chain,
                device=device,
                batch_size=params.batch_size,
                seq_len=params.seq_len,
                seed=seed,
                topk=expectation_topk,
            )
        return markov_regression_mse_curve(
            model=model,
            snr_grid=snr_grid,
            chain=self.chain,
            device=device,
            batch_size=params.batch_size,
            seq_len=params.seq_len,
            seed=seed,
        )

    def metric_fields(self) -> dict[str, Any]:
        return {
            "alphabet_size": int(self.alphabet_size),
            "dirichlet_alpha": float(self.dirichlet_alpha),
            "init_distribution": self.init_distribution,
            "entropy_rate_ground_truth_nats": self.reference_entropy_nats,
        }

def _format_tensor_preview(tensor: torch.Tensor, max_items: int = 6) -> str:
    flat = torch.as_tensor(tensor, dtype=torch.float64).detach().cpu().reshape(-1)
    if flat.numel() <= max_items:
        return "[" + ", ".join(f"{float(value):.4f}" for value in flat.tolist()) + "]"
    head_count = max(1, max_items // 2)
    tail_count = max(1, max_items - head_count)
    head = ", ".join(f"{float(value):.4f}" for value in flat[:head_count].tolist())
    tail = ", ".join(f"{float(value):.4f}" for value in flat[-tail_count:].tolist())
    return f"[{head}, ..., {tail}]"
