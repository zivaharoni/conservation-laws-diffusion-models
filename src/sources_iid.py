"""IID source sampling helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from .channels import sample_log_uniform_snr
from .data_common import EvaluationParams, TrainingBatch


def uniform_iid_probs(alphabet_size: int) -> torch.Tensor:
    return torch.full((alphabet_size,), 1.0 / float(alphabet_size), dtype=torch.float64)


def sample_dirichlet_iid_probs(alphabet_size: int, dirichlet_alpha: float, seed: int) -> torch.Tensor:
    concentration = torch.full((alphabet_size,), float(dirichlet_alpha), dtype=torch.float64)
    rng_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(int(seed))
        gamma = torch.distributions.Gamma(concentration, torch.ones_like(concentration)).sample()
    finally:
        torch.random.set_rng_state(rng_state)
    return gamma / gamma.sum()


def entropy_nats(probs: torch.Tensor) -> float:
    probs_t = torch.as_tensor(probs, dtype=torch.float64)
    probs_t = torch.clamp(probs_t / probs_t.sum(), min=1e-12)
    return float(-(probs_t * torch.log(probs_t)).sum().item())


def log_priors(
    probs: torch.Tensor,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    probs_t = torch.as_tensor(probs, device=device, dtype=dtype or torch.float32)
    probs_t = probs_t / probs_t.sum()
    return torch.log(torch.clamp(probs_t, min=1e-12))


def sample_iid_states(batch_size: int, probs: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    probs_t = torch.as_tensor(probs, device=device, dtype=torch.float32)
    probs_t = probs_t / probs_t.sum()
    return torch.multinomial(probs_t, num_samples=batch_size, replacement=True)


@dataclass(slots=True)
class IidSourceData:
    alphabet_size: int
    seed: int
    source_distribution: str = "uniform"
    dirichlet_alpha: float | None = None
    source_probs: torch.Tensor = field(init=False)
    title: str = field(init=False)
    reference_label: str = field(init=False)
    reference_unit: str = field(init=False)
    run_label: str = field(init=False)

    def __post_init__(self) -> None:
        self.source_distribution = str(self.source_distribution).lower()
        if self.source_distribution == "dirichlet":
            if self.dirichlet_alpha is None:
                raise ValueError("dirichlet_alpha must be set when source_distribution='dirichlet'.")
            self.source_probs = sample_dirichlet_iid_probs(self.alphabet_size, float(self.dirichlet_alpha), int(self.seed))
        elif self.source_distribution == "uniform":
            self.source_probs = uniform_iid_probs(self.alphabet_size)
        else:
            raise ValueError(
                f"Unsupported source_distribution={self.source_distribution!r}; expected 'uniform' or 'dirichlet'."
            )
        self.title = "IID MMSE Curves"
        self.reference_label = "Entropy ground truth"
        self.reference_unit = "nats"
        self.run_label = "iid"

    @property
    def reference_entropy_nats(self) -> float:
        return entropy_nats(self.source_probs)

    def data_attributes(self) -> list[tuple[str, str]]:
        attrs = [
            ("alphabet_size", str(int(self.alphabet_size))),
            ("source_distribution", self.source_distribution),
            ("source_probs", _format_tensor_preview(self.source_probs)),
        ]
        if self.source_distribution == "dirichlet":
            attrs.insert(2, ("dirichlet_alpha", f"{float(self.dirichlet_alpha):.4f}"))
        return attrs

    def model_init_kwargs(self) -> dict[str, object]:
        return {"alphabet_size": int(self.alphabet_size)}

    def dummy_forward_inputs(
        self,
        model: torch.nn.Module,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = 2
        embed_dim = int(model.embed_dim)
        y = torch.zeros((batch_size, embed_dim), device=device, dtype=dtype)
        snr = torch.ones((batch_size, 1), device=device, dtype=dtype)
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
        del seed
        probs_t = torch.as_tensor(self.source_probs, dtype=torch.float32, device=device)

        def _sample() -> TrainingBatch:
            snr = sample_log_uniform_snr((batch_size, 1), snr_min, snr_max, device=device)
            states = sample_iid_states(batch_size, probs_t, device=device)
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
        from .immse import iid_mmse_curves

        return iid_mmse_curves(
            model=model,
            snr_grid=snr_grid,
            source_probs=self.source_probs,
            device=device,
            n_samples=params.n_samples,
            batch_size=params.batch_size,
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
        from .immse import iid_model_mmse_curve, iid_regression_mse_curve

        if expectation_method == "probs":
            return iid_model_mmse_curve(
                model=model,
                snr_grid=snr_grid,
                source_probs=self.source_probs,
                device=device,
                n_samples=params.n_samples,
                batch_size=params.batch_size,
                seed=seed,
                topk=expectation_topk,
            )
        return iid_regression_mse_curve(
            model=model,
            snr_grid=snr_grid,
            source_probs=self.source_probs,
            device=device,
            n_samples=params.n_samples,
            batch_size=params.batch_size,
            seed=seed,
        )

    def metric_fields(self) -> dict[str, Any]:
        metrics = {
            "alphabet_size": int(self.alphabet_size),
            "entropy_ground_truth_nats": self.reference_entropy_nats,
            "source_distribution": self.source_distribution,
        }
        if self.source_distribution == "dirichlet":
            metrics["dirichlet_alpha"] = float(self.dirichlet_alpha)
        return metrics


def _format_tensor_preview(tensor: torch.Tensor, max_items: int = 6) -> str:
    flat = torch.as_tensor(tensor, dtype=torch.float64).detach().cpu().reshape(-1)
    if flat.numel() <= max_items:
        return "[" + ", ".join(f"{float(value):.4f}" for value in flat.tolist()) + "]"
    head_count = max(1, max_items // 2)
    tail_count = max(1, max_items - head_count)
    head = ", ".join(f"{float(value):.4f}" for value in flat[:head_count].tolist())
    tail = ", ".join(f"{float(value):.4f}" for value in flat[-tail_count:].tolist())
    return f"[{head}, ..., {tail}]"
