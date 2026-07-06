"""Evaluation, plotting, and metric assembly helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig

from src.data_common import EvaluationParams
from src.immse import integrate_half_mmse_curve, relative_error

from .experiment_registry import require_experiment_kind
from .runtime import eval_expectation_method, eval_expectation_topk, loss_estimator, loss_name, loss_topk


def save_mmse_curve_plot(
    output_path: Path,
    snr_grid: torch.Tensor,
    true_mmse: torch.Tensor,
    model_mmse: torch.Tensor,
    true_area: float,
    model_area: float,
    title: str,
) -> None:
    snr_grid = torch.as_tensor(snr_grid, dtype=torch.float64).cpu()
    true_mmse = torch.as_tensor(true_mmse, dtype=torch.float64).cpu()
    model_mmse = torch.as_tensor(model_mmse, dtype=torch.float64).cpu()
    true_half_mmse = 0.5 * true_mmse
    model_half_mmse = 0.5 * model_mmse
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(snr_grid, true_half_mmse, linewidth=2.4, color="tab:blue", label=f"Reference area={true_area:.4f}")
    ax.plot(
        snr_grid,
        model_half_mmse,
        linestyle="-.",
        linewidth=2.0,
        color="tab:orange",
        label=f"Learned model area={model_area:.4f}",
    )
    ax.fill_between(snr_grid, torch.zeros_like(snr_grid), true_half_mmse, color="tab:blue", alpha=0.22)
    ax.fill_between(snr_grid, torch.zeros_like(snr_grid), model_half_mmse, color="tab:orange", alpha=0.18)
    ax.set_xscale("log")
    ax.set_xlabel("SNR gamma")
    ax.set_ylabel("0.5 * MMSE(gamma)")
    ax.set_title(title)
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def evaluate_model(
    cfg: DictConfig,
    model: torch.nn.Module,
    device: torch.device,
    embed_dim: int,
    data_spec,
    eval_overrides: object | None = None,
) -> tuple[torch.Tensor, dict, dict]:
    require_experiment_kind(cfg, "immse")
    params = _resolve_eval_params(cfg, eval_overrides)
    snr_grid = torch.logspace(
        torch.log10(torch.tensor(float(cfg.snr_min), dtype=torch.float64)),
        torch.log10(torch.tensor(float(cfg.snr_max), dtype=torch.float64)),
        params.n_snr_points,
        dtype=torch.float64,
    )
    eval_method = eval_expectation_method(cfg)
    eval_topk = eval_expectation_topk(cfg)

    reference_curves = data_spec.reference_curves(
        model=model,
        snr_grid=snr_grid,
        device=device,
        params=params,
        seed=cfg.seed + 500,
    )
    model_curve = data_spec.model_curve(
        model=model,
        snr_grid=snr_grid,
        device=device,
        params=params,
        expectation_method=eval_method,
        expectation_topk=eval_topk,
        seed=cfg.seed + 1_500,
    )

    curves = {"true_mmse": reference_curves["true_mmse"], "model_mmse": model_curve}
    reference_entropy = data_spec.reference_entropy_nats
    reference_area = integrate_half_mmse_curve(snr_grid, curves["true_mmse"])
    model_area = integrate_half_mmse_curve(snr_grid, curves["model_mmse"])

    metrics = {
        "alphabet_size": int(data_spec.alphabet_size),
        "embed_dim": int(embed_dim),
        "loss_name": loss_name(cfg),
        "loss_estimator": loss_estimator(cfg),
        "loss_topk": loss_topk(cfg),
        "eval_expectation_method": eval_method,
        "eval_expectation_topk": eval_topk,
        "reference_entropy_nats": reference_entropy,
        "reference_area_nats": reference_area,
        "reference_abs_error": abs(reference_area - reference_entropy),
        "reference_rel_error": relative_error(reference_area, reference_entropy),
        "model_area_nats": model_area,
        "model_abs_error": abs(model_area - reference_entropy),
        "model_rel_error": relative_error(model_area, reference_entropy),
        **data_spec.metric_fields(),
    }
    summary = {
        "title": data_spec.title,
        "reference_label": data_spec.reference_label,
        "reference_unit": data_spec.reference_unit,
    }
    return snr_grid, curves, {**metrics, **summary}


def _resolve_eval_params(cfg: DictConfig, overrides: object | None = None) -> EvaluationParams:
    def _value(name: str) -> int:
        if overrides is not None and hasattr(overrides, name):
            value = getattr(overrides, name)
            if value is not None:
                return int(value)
        require_experiment_kind(cfg, "immse")
        return int(getattr(cfg.eval, name))

    return EvaluationParams(
        n_snr_points=_value("n_snr_points"),
        n_samples=_value("n_samples"),
        batch_size=_value("batch_size"),
        seq_len=_value("seq_len"),
    )
