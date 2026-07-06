from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from experiments.common.io import ensure_dir, save_csv, save_json
from experiments.common.repro import configure_environment, get_best_device, set_global_seed
from scripts.eval_img_generation_checkpoints import (
    _resolve_run_path,
    build_image_model,
    checkpoint_step,
    compute_generation_metrics,
    sample_from_logits,
)
from scripts.img_common import find_repo_root, load_img_benchmark
from src.channels import apply_ksc_noise, awgn, log_ksc_kernel
from src.lm_distributed import unwrap_model
from src.lm_objectives import apply_sequence_masking, evaluate_exit_curve, evaluate_immse_curve, evaluate_qsc_curve
from src.models import posterior_mean_from_logits


DEFAULT_STEP_BUDGETS = (25, 50, 100, 200, 400, 1000, 4000)

def _cfg_string(value: Any) -> str:
    """Normalize Hydra/OmegaConf scalars that may be wrapped like {'value': '...'}."""
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    container = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    if isinstance(container, dict) and "value" in container and len(container) == 1:
        return str(container["value"])
    return str(container)


def _cfg_unwrap(value: Any) -> Any:
    """Unwrap Hydra/OmegaConf containers that may be wrapped like {'value': {...}}."""
    if value is None:
        return None
    if OmegaConf.is_config(value):
        container = OmegaConf.to_container(value, resolve=True)
    else:
        container = value
    if isinstance(container, dict) and "value" in container and len(container) == 1:
        return container["value"]
    return container


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full CIFAR generation/conservation suite for final and EMA weights.")
    parser.add_argument("--run", required=True, help="W&B run path entity/project/run_id, or a bare run id.")
    parser.add_argument("--entity", default=None, help="W&B entity for bare run ids.")
    parser.add_argument("--project", default=None, help="W&B project for bare run ids.")
    parser.add_argument("--cache-dir", default=None, help="Directory containing or receiving downloaded config/checkpoints.")
    parser.add_argument("--output-dir", default=None, help="Directory for suite outputs.")
    parser.add_argument("--sample-count", type=int, default=50_000, help="Generated image count per metric setting.")
    parser.add_argument("--split", default="test", help="Real-image split for conservation and FID.")
    parser.add_argument("--seed", type=int, default=2026, help="Base deterministic seed.")
    parser.add_argument("--device", default="auto", help="Torch device, or auto.")
    parser.add_argument("--batch-size", type=int, default=None, help="Generation/metric batch size override.")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Conservation-law eval batch size override.")
    parser.add_argument("--ema-only", action="store_true", help="Evaluate only EMA weights (ignores --weight-variants).")
    parser.add_argument("--conservation-points", type=int, default=100, help="Number of evaluated conservation-law knots.")
    parser.add_argument("--conservation-steps-per-batch", type=int, default=1, help="Noise draws per test batch at each conservation-law knot.")
    parser.add_argument("--step-budgets", default=",".join(str(value) for value in DEFAULT_STEP_BUDGETS), help="Comma-separated generation step budgets.")
    parser.add_argument("--weight-variants", default="final,ema", help="Comma-separated weight variants to evaluate: final,ema.")
    parser.add_argument("--schedules", default="uniform,equal_info", help="Comma-separated schedules to evaluate: uniform,equal_info,cosine,derivative.")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional max eval batches for smoke tests. Default uses the full split.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for discrete posterior logits.")
    parser.add_argument("--top-k", type=int, default=0, help="Optional top-k posterior truncation; 0 disables it.")
    parser.add_argument("--grid-rows", type=int, default=3, help="Rows for a generated-image grid (saved when --grid-steps matches a budget).")
    parser.add_argument("--grid-cols", type=int, default=10, help="Columns for a generated-image grid (saved when --grid-steps matches a budget).")
    parser.add_argument("--grid-steps", type=int, default=1000, help="Which diffusion-step budget to save generated-image grids for; 0 disables.")
    parser.add_argument("--trace-count", type=int, default=1, help="How many independent generation traces to capture per selected step budget.")
    parser.add_argument("--trace-frames", type=int, default=10, help="How many frames to capture per trace (evenly spaced indices).")
    parser.add_argument("--trace-steps", type=int, default=1000, help="Which diffusion-step budget to save traces for; 0 means all budgets.")
    parser.add_argument("--auto-batch", action="store_true", help="Probe the largest generation batch size that fits on this GPU.")
    parser.add_argument("--auto-eval-batch", action="store_true", help="Probe the largest conservation-law eval batch size that fits on this GPU.")
    parser.add_argument("--auto-batch-max", type=int, default=4096, help="Maximum batch size considered by --auto-batch/--auto-eval-batch.")
    parser.add_argument("--log-wandb", action="store_true", help="Log suite metrics, figures, and tables to a new W&B run.")
    parser.add_argument("--wandb-name", default=None, help="Optional W&B run name for the eval suite.")
    parser.add_argument("--wandb-group", default=None, help="Optional W&B group for the eval suite.")
    parser.add_argument("--progress-log-every", type=int, default=0, help="When W&B logging is enabled, log generated-sample progress every N samples. 0 disables progress logs.")
    parser.add_argument("--progress-print-every", type=int, default=1000, help="Print sample-generation progress every N samples (for Slurm logs). 0 disables printing.")
    parser.add_argument("--no-progress-bar", action="store_true", help="Disable progress bar when running interactively.")
    return parser


def _safe_run_id(run: str) -> str:
    return str(run).rstrip("/").split("/")[-1]


def _default_cache_dir(run: str) -> Path:
    return APP_ROOT / "results" / "img" / "cifar_suite" / _safe_run_id(run) / "downloads"


def _default_output_dir(run: str) -> Path:
    return APP_ROOT / "results" / "img" / "cifar_suite" / _safe_run_id(run)


def _parse_step_budgets(value: str) -> list[int]:
    budgets = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not budgets or any(step <= 0 for step in budgets):
        raise ValueError("--step-budgets must contain positive integers.")
    return budgets


def _parse_choices(value: str, *, allowed: set[str], arg_name: str) -> list[str]:
    choices = [part.strip() for part in str(value).split(",") if part.strip()]
    if not choices:
        raise ValueError(f"{arg_name} must contain at least one value.")
    unknown = sorted(set(choices) - set(allowed))
    if unknown:
        raise ValueError(f"{arg_name} contains unsupported values: {unknown}; allowed={sorted(allowed)}.")
    return choices


def _normalize_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {key[len("module."):] if key.startswith("module.") else key: value for key, value in state_dict.items()}


def load_final_and_ema_state_dicts(path: str | Path, *, required_variants: list[str] | None = None) -> dict[str, dict[str, Any]]:
    required = set(required_variants or ["final", "ema"])
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint at {path} must deserialize to a dict payload.")
    model_state = payload.get("model_state_dict")
    ema_state = payload.get("ema_state_dict")
    if not isinstance(model_state, dict):
        raise KeyError(f"Checkpoint {path} does not contain model_state_dict.")
    if "ema" in required and not isinstance(ema_state, dict):
        raise KeyError(f"Checkpoint {path} does not contain ema_state_dict; EMA is required for this suite.")
    states = {"final": _normalize_state_dict_keys(model_state)}
    if isinstance(ema_state, dict):
        states["ema"] = _normalize_state_dict_keys(ema_state)
    return states


def discover_cached_final_checkpoint(cache_dir: str | Path) -> Path | None:
    checkpoints = sorted(Path(cache_dir).glob("checkpoints/model_step_*.pt"), key=checkpoint_step)
    return checkpoints[-1] if checkpoints else None


def ensure_cached_config_and_final_checkpoint(*, run: str, cache_dir: Path, entity: str | None, project: str | None) -> tuple[Path, Path]:
    config_path = cache_dir / "config.yaml"
    cached_checkpoint = discover_cached_final_checkpoint(cache_dir)
    if config_path.is_file() and cached_checkpoint is not None:
        return config_path, cached_checkpoint

    import wandb

    api = wandb.Api()
    cached_cfg = OmegaConf.load(config_path) if config_path.is_file() else None
    run_path = _resolve_run_path(run, cfg=cached_cfg, entity=entity, project=project)
    wb_run = api.run(run_path)
    files = list(wb_run.files())
    if not config_path.is_file():
        config_file = next((file_obj for file_obj in files if str(file_obj.name) == "config.yaml"), None)
        if config_file is None:
            raise FileNotFoundError(f"W&B run {run_path} does not contain config.yaml.")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_file.download(root=str(cache_dir), replace=False)

    cached_checkpoint = discover_cached_final_checkpoint(cache_dir)
    checkpoint_files = [file_obj for file_obj in files if str(file_obj.name).startswith("checkpoints/model_step_") and str(file_obj.name).endswith(".pt")]
    if not checkpoint_files and cached_checkpoint is None:
        raise FileNotFoundError(f"W&B run {run_path} does not contain checkpoints/model_step_*.pt.")
    if checkpoint_files:
        final_file = max(checkpoint_files, key=lambda file_obj: checkpoint_step(str(file_obj.name)))
        final_path = cache_dir / str(final_file.name)
        if not final_path.is_file():
            final_path.parent.mkdir(parents=True, exist_ok=True)
            final_file.download(root=str(cache_dir), replace=False)
        cached_checkpoint = final_path
    assert cached_checkpoint is not None
    return config_path, cached_checkpoint


def _make_model(cfg: Any, benchmark: Any, state_dict: dict[str, Any], device: torch.device) -> torch.nn.Module:
    model = build_image_model(cfg, vocab_size=benchmark.vocab_size, context_length=benchmark.context_length)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def _variant_seed(base_seed: int, *, variant: str, num_steps: int) -> int:
    offset = 0 if variant == "final" else 1_000_000
    return int(base_seed) + offset + int(num_steps)


def _capture_indices(num_steps: int, count: int = 10) -> set[int]:
    values = torch.linspace(0, int(num_steps), steps=int(count), dtype=torch.float64)
    return {int(round(float(value))) for value in values.tolist()}


def _decode_nearest_embeddings(model: torch.nn.Module, benchmark: Any, embeddings: torch.Tensor) -> torch.Tensor:
    base_model = unwrap_model(model)
    distances = torch.cdist(embeddings.reshape(-1, embeddings.shape[-1]).cpu(), base_model.x_values.weight.detach().cpu())
    tokens = torch.argmin(distances, dim=1).view(int(embeddings.shape[0]), int(embeddings.shape[1]))
    return benchmark.decode_token_sequences(tokens)


def _decode_continuous_values(model: torch.nn.Module, benchmark: Any, values: torch.Tensor) -> torch.Tensor:
    base_model = unwrap_model(model)
    if hasattr(base_model, "dequantize_mean_to_rgb_tokens"):
        tokens = base_model.dequantize_mean_to_rgb_tokens(values.detach().cpu()).view(int(values.shape[0]), int(values.shape[1]))
        return benchmark.decode_token_sequences(tokens)
    return _decode_nearest_embeddings(model, benchmark, values)


@torch.no_grad()
def _exit_reverse_mask_transition(
    *,
    sampled_clean: torch.Tensor,
    observed: torch.Tensor,
    current_s: torch.Tensor,
    next_s: torch.Tensor,
    mask_token_id: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Sample q(x_next | x_current, x0) for the masking channel.

    Tokens revealed at the current noise level must stay revealed. Only tokens
    that are still masked can remain masked, with probability next_s/current_s.
    """
    if sampled_clean.shape != observed.shape:
        raise ValueError("sampled_clean and observed must have the same shape.")
    batch_size, seq_len = observed.shape
    current = torch.as_tensor(current_s, dtype=torch.float32, device=observed.device).view(batch_size, 1)
    nxt = torch.as_tensor(next_s, dtype=torch.float32, device=observed.device).view(batch_size, 1)
    remain_mask_prob = torch.where(current > 1.0e-12, torch.clamp(nxt / current, min=0.0, max=1.0), torch.zeros_like(current))
    still_masked = observed == int(mask_token_id)
    if generator is None:
        uniforms = torch.rand((batch_size, seq_len), dtype=torch.float32, device=observed.device)
    elif torch.device(generator.device).type == observed.device.type:
        uniforms = torch.rand((batch_size, seq_len), generator=generator, dtype=torch.float32, device=observed.device)
    else:
        uniforms = torch.rand((batch_size, seq_len), generator=generator, dtype=torch.float32).to(device=observed.device)
    keep_mask = still_masked & (uniforms < remain_mask_prob)
    next_observed = torch.where(still_masked, sampled_clean, observed)
    next_observed = torch.where(keep_mask, torch.full_like(next_observed, int(mask_token_id)), next_observed)
    return next_observed


@torch.no_grad()
def sample_exit_with_schedule(
    model: torch.nn.Module,
    benchmark: Any,
    *,
    schedule: torch.Tensor,
    count: int,
    batch_size: int,
    temperature: float,
    top_k: int,
    device: torch.device,
    generator: torch.Generator,
    trace: bool = False,
    trace_frame_count: int = 10,
    progress_callback: Any | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    base_model = unwrap_model(model)
    generated: list[torch.Tensor] = []
    trace_frames: list[torch.Tensor] = []
    capture = _capture_indices(int(schedule.numel()) - 1, count=int(trace_frame_count)) if trace else set()
    while sum(int(chunk.shape[0]) for chunk in generated) < int(count):
        current_batch = min(int(batch_size), int(count) - sum(int(chunk.shape[0]) for chunk in generated))
        capture_this_batch = bool(trace) and not generated
        clean = torch.full((current_batch, benchmark.context_length), int(base_model.mask_token_id), dtype=torch.long, device=device)
        observed = clean
        if capture_this_batch and 0 in capture:
            trace_frames.append(benchmark.decode_token_sequences(observed[:1].cpu(), mask_token_id=int(base_model.mask_token_id))[0])
        for step_idx in range(int(schedule.numel()) - 1):
            s = schedule[step_idx].expand(current_batch).to(device=device)
            next_s = schedule[step_idx + 1].expand(current_batch).to(device=device)
            logits = model(observed, timesteps=s)
            clean = sample_from_logits(logits, temperature=temperature, top_k=top_k, generator=generator)
            observed = _exit_reverse_mask_transition(
                sampled_clean=clean,
                observed=observed,
                current_s=s,
                next_s=next_s,
                mask_token_id=int(base_model.mask_token_id),
                generator=generator,
            )
            if capture_this_batch and (step_idx + 1) in capture:
                trace_frames.append(benchmark.decode_token_sequences(observed[:1].cpu(), mask_token_id=int(base_model.mask_token_id))[0])
        final_tokens = torch.where(observed == int(base_model.mask_token_id), clean, observed)
        generated.append(benchmark.decode_token_sequences(final_tokens.cpu()))
        if progress_callback is not None:
            progress_callback(min(sum(int(chunk.shape[0]) for chunk in generated), int(count)))
    return torch.cat(generated, dim=0)[: int(count)].clamp(0.0, 1.0), trace_frames


@torch.no_grad()
def sample_qsc_with_schedule(
    model: torch.nn.Module,
    benchmark: Any,
    *,
    schedule: torch.Tensor,
    count: int,
    batch_size: int,
    temperature: float,
    top_k: int,
    device: torch.device,
    generator: torch.Generator,
    trace: bool = False,
    trace_frame_count: int = 10,
    progress_callback: Any | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    base_model = unwrap_model(model)
    generated: list[torch.Tensor] = []
    trace_frames: list[torch.Tensor] = []
    capture = _capture_indices(int(schedule.numel()) - 1, count=int(trace_frame_count)) if trace else set()
    while sum(int(chunk.shape[0]) for chunk in generated) < int(count):
        current_batch = min(int(batch_size), int(count) - sum(int(chunk.shape[0]) for chunk in generated))
        capture_this_batch = bool(trace) and not generated
        clean = torch.randint(int(base_model.vocab_size), (current_batch, benchmark.context_length), generator=generator, dtype=torch.long).to(device=device)
        observed = clean
        if capture_this_batch and 0 in capture:
            trace_frames.append(benchmark.decode_token_sequences(observed[:1].cpu())[0])
        for step_idx in range(int(schedule.numel()) - 1):
            t = schedule[step_idx].expand(current_batch).to(device=device)
            next_t = schedule[step_idx + 1].expand(current_batch).to(device=device)
            kernel = log_ksc_kernel(observed, t, int(base_model.vocab_size))
            outputs = model.forward_outputs(observed, timesteps=t, channel_tilt=kernel) if hasattr(model, "forward_outputs") else None
            logits = outputs.logits if outputs is not None else model(observed, timesteps=t) + kernel
            clean = sample_from_logits(logits, temperature=temperature, top_k=top_k, generator=generator)
            observed, _ = apply_ksc_noise(clean, next_t, int(base_model.vocab_size), generator=generator)
            if capture_this_batch and (step_idx + 1) in capture:
                trace_frames.append(benchmark.decode_token_sequences(clean[:1].cpu())[0])
        generated.append(benchmark.decode_token_sequences(clean.cpu()))
        if progress_callback is not None:
            progress_callback(min(sum(int(chunk.shape[0]) for chunk in generated), int(count)))
    return torch.cat(generated, dim=0)[: int(count)].clamp(0.0, 1.0), trace_frames


@torch.no_grad()
def sample_immse_with_schedule(
    model: torch.nn.Module,
    benchmark: Any,
    *,
    schedule: torch.Tensor,
    count: int,
    batch_size: int,
    temperature: float,
    top_k: int,
    device: torch.device,
    generator: torch.Generator,
    trace: bool = False,
    trace_frame_count: int = 10,
    progress_callback: Any | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    base_model = unwrap_model(model)
    generated: list[torch.Tensor] = []
    trace_frames: list[torch.Tensor] = []
    capture = _capture_indices(int(schedule.numel()) - 1, count=int(trace_frame_count)) if trace else set()
    while sum(int(chunk.shape[0]) for chunk in generated) < int(count):
        current_batch = min(int(batch_size), int(count) - sum(int(chunk.shape[0]) for chunk in generated))
        capture_this_batch = bool(trace) and not generated
        snr0 = schedule[0].expand(current_batch, benchmark.context_length, 1).to(device=device)
        noisy_shape = (current_batch, benchmark.context_length, int(base_model.x_values.embedding_dim))
        if generator is None or torch.device(generator.device).type == device.type:
            noisy = torch.randn(noisy_shape, generator=generator, dtype=torch.float32, device=device)
        else:
            noisy = torch.randn(noisy_shape, generator=generator, dtype=torch.float32).to(device=device)
        noisy = noisy / torch.sqrt(torch.clamp(snr0, min=1.0e-12))
        tokens = None
        if capture_this_batch and 0 in capture:
            trace_frames.append(_decode_nearest_embeddings(model, benchmark, noisy[:1])[0])
        for step_idx in range(int(schedule.numel()) - 1):
            snr = schedule[step_idx].expand(current_batch, benchmark.context_length, 1).to(device=device)
            next_snr = schedule[step_idx + 1].expand(current_batch, benchmark.context_length, 1).to(device=device)
            emb_hat, _logits_ext, logits = model(noisy, snr)
            mean = posterior_mean_from_logits(logits, base_model.x_values.weight.to(device=device, dtype=noisy.dtype)) if emb_hat is None else emb_hat
            denoised = emb_hat if emb_hat is not None else mean
            log_snr_step = torch.log(torch.clamp(next_snr, min=1.0e-12)) - torch.log(torch.clamp(snr, min=1.0e-12))
            noisy = noisy + 0.5 * log_snr_step * (denoised - noisy)
            if capture_this_batch and (step_idx + 1) in capture:
                trace_frames.append(_decode_continuous_values(model, benchmark, denoised[:1])[0])
        final_snr = schedule[-1].expand(current_batch, benchmark.context_length, 1).to(device=device)
        _final_emb_hat, _final_logits_ext, final_logits = model(noisy, final_snr)
        tokens = sample_from_logits(final_logits, temperature=temperature, top_k=top_k, generator=generator)
        generated.append(benchmark.decode_token_sequences(tokens.cpu()))
        if progress_callback is not None:
            progress_callback(min(sum(int(chunk.shape[0]) for chunk in generated), int(count)))
    return torch.cat(generated, dim=0)[: int(count)].clamp(0.0, 1.0), trace_frames


def sample_with_schedule(model: torch.nn.Module, benchmark: Any, cfg: Any, **kwargs: Any) -> tuple[torch.Tensor, list[torch.Tensor]]:
    kind = str(cfg.kind)
    if kind == "img_exit":
        return sample_exit_with_schedule(model, benchmark, **kwargs)
    if kind == "img_qsc":
        return sample_qsc_with_schedule(model, benchmark, **kwargs)
    if kind == "img_immse":
        return sample_immse_with_schedule(model, benchmark, **kwargs)
    raise ValueError(f"Unsupported CIFAR suite kind={kind!r}; expected img_exit, img_qsc, or img_immse.")


def _curve_coordinate_and_density(cfg: Any, grid: torch.Tensor, curve: Any) -> tuple[torch.Tensor, torch.Tensor, str, bool]:
    kind = str(cfg.kind)
    if kind == "img_immse":
        snr = torch.as_tensor(grid, dtype=torch.float64)
        mmse = torch.as_tensor(curve, dtype=torch.float64)
        return torch.log(snr), 0.5 * mmse * snr, "snr", True
    if kind == "img_exit":
        curve_dict = curve
        return torch.as_tensor(grid, dtype=torch.float64), torch.as_tensor(curve_dict["exit_gap"], dtype=torch.float64), "s", False
    if kind == "img_qsc":
        curve_dict = curve
        return torch.as_tensor(grid, dtype=torch.float64), torch.as_tensor(curve_dict["qsc_gexit"], dtype=torch.float64), "t", False
    raise ValueError(f"Unsupported kind={kind!r}.")


def _native_schedule_from_area_coordinate(cfg: Any, x: torch.Tensor) -> torch.Tensor:
    if str(cfg.kind) == "img_immse":
        return torch.exp(x)
    return x


def build_uniform_schedule(cfg: Any, *, num_steps: int) -> torch.Tensor:
    kind = str(cfg.kind)
    if kind == "img_exit":
        return torch.linspace(1.0, 0.0, steps=int(num_steps) + 1, dtype=torch.float32)
    if kind == "img_qsc":
        t_min = float(getattr(cfg.eval, "t_min", getattr(cfg.qsc, "train_t_min", 1.0 / float(cfg.benchmark.codebook_size))))
        t_max = float(getattr(cfg.eval, "t_max", getattr(cfg.qsc, "train_t_max", 0.9999)))
        return torch.linspace(t_min, t_max, steps=int(num_steps) + 1, dtype=torch.float32)
    if kind == "img_immse":
        snr_min = float(getattr(cfg.eval, "snr_min", getattr(cfg.train, "snr_min", 1.0e-6)))
        snr_max = float(getattr(cfg.eval, "snr_max", getattr(cfg.train, "snr_max", 1.0)))
        return torch.exp(torch.linspace(math.log(snr_min), math.log(snr_max), steps=int(num_steps) + 1, dtype=torch.float32))
    raise ValueError(f"Unsupported kind={kind!r}.")


def build_cosine_schedule(cfg: Any, *, num_steps: int) -> torch.Tensor:
    u = torch.linspace(0.0, 1.0, steps=int(num_steps) + 1, dtype=torch.float64)
    cosine = 0.5 * (1.0 + torch.cos(math.pi * u))
    kind = str(cfg.kind)
    if kind == "img_exit":
        return cosine.to(dtype=torch.float32)
    if kind == "img_qsc":
        t_min = float(getattr(cfg.eval, "t_min", getattr(cfg.qsc, "train_t_min", 1.0 / float(cfg.benchmark.codebook_size))))
        t_max = float(getattr(cfg.eval, "t_max", getattr(cfg.qsc, "train_t_max", 0.9999)))
        return (t_min + (t_max - t_min) * (1.0 - cosine)).to(dtype=torch.float32)
    if kind == "img_immse":
        snr_min = float(getattr(cfg.eval, "snr_min", getattr(cfg.train, "snr_min", 1.0e-6)))
        snr_max = float(getattr(cfg.eval, "snr_max", getattr(cfg.train, "snr_max", 1.0)))
        log_snr = math.log(snr_min) + (math.log(snr_max) - math.log(snr_min)) * (1.0 - cosine)
        return torch.exp(log_snr).to(dtype=torch.float32)
    raise ValueError(f"Unsupported kind={kind!r}.")


def build_equal_info_schedule(cfg: Any, *, area_x: torch.Tensor, density: torch.Tensor, num_steps: int) -> torch.Tensor:
    x = torch.as_tensor(area_x, dtype=torch.float64)
    y = torch.nan_to_num(torch.as_tensor(density, dtype=torch.float64), nan=0.0, posinf=0.0, neginf=0.0)
    y = torch.clamp(y, min=0.0)
    order = torch.argsort(x)
    x = x[order]
    y = y[order]
    segment_area = 0.5 * (y[:-1] + y[1:]) * torch.clamp(x[1:] - x[:-1], min=0.0)
    cumulative = torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum(segment_area, dim=0)])
    total = float(cumulative[-1].item())
    if not math.isfinite(total) or total <= 1.0e-12:
        schedule = build_uniform_schedule(cfg, num_steps=num_steps).to(dtype=torch.float64)
        return schedule
    keep = torch.ones_like(cumulative, dtype=torch.bool)
    keep[1:] = cumulative[1:] > cumulative[:-1]
    cumulative = cumulative[keep] / total
    x = x[keep]
    targets = torch.linspace(0.0, 1.0, steps=int(num_steps) + 1, dtype=torch.float64)
    idx = torch.searchsorted(cumulative, targets, right=False).clamp(max=int(cumulative.numel()) - 1)
    left_idx = torch.clamp(idx - 1, min=0)
    right_idx = idx
    c0 = cumulative[left_idx]
    c1 = cumulative[right_idx]
    x0 = x[left_idx]
    x1 = x[right_idx]
    frac = torch.where((c1 - c0).abs() > 1.0e-12, (targets - c0) / (c1 - c0), torch.zeros_like(targets))
    native = _native_schedule_from_area_coordinate(cfg, x0 + frac * (x1 - x0))
    if str(cfg.kind) == "img_exit":
        native = torch.flip(native, dims=(0,))
    return native.to(dtype=torch.float32)


def build_derivative_schedule(cfg: Any, *, area_x: torch.Tensor, density: torch.Tensor, num_steps: int) -> torch.Tensor:
    x = torch.as_tensor(area_x, dtype=torch.float64)
    y = torch.as_tensor(density, dtype=torch.float64)
    order = torch.argsort(x)
    x = x[order]
    y = y[order]
    finite = torch.isfinite(x) & torch.isfinite(y)
    x = x[finite]
    y = y[finite]
    if int(x.numel()) < 2:
        return build_uniform_schedule(cfg, num_steps=num_steps)
    dx = torch.clamp(x[1:] - x[:-1], min=0.0)
    slope_abs = torch.where(dx > 0.0, (y[1:] - y[:-1]).abs() / torch.clamp(dx, min=1.0e-18), torch.zeros_like(dx))
    segment_area = torch.sqrt(torch.nan_to_num(slope_abs, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)) * dx
    cumulative = torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum(segment_area, dim=0)])
    total = float(cumulative[-1].item())
    if not math.isfinite(total) or total <= 1.0e-12:
        return build_uniform_schedule(cfg, num_steps=num_steps)
    keep = torch.ones_like(cumulative, dtype=torch.bool)
    keep[1:] = cumulative[1:] > cumulative[:-1]
    cumulative = cumulative[keep] / total
    x = x[keep]
    targets = torch.linspace(0.0, 1.0, steps=int(num_steps) + 1, dtype=torch.float64)
    idx = torch.searchsorted(cumulative, targets, right=False).clamp(max=int(cumulative.numel()) - 1)
    left_idx = torch.clamp(idx - 1, min=0)
    right_idx = idx
    c0 = cumulative[left_idx]
    c1 = cumulative[right_idx]
    x0 = x[left_idx]
    x1 = x[right_idx]
    frac = torch.where((c1 - c0).abs() > 1.0e-12, (targets - c0) / (c1 - c0), torch.zeros_like(targets))
    native = _native_schedule_from_area_coordinate(cfg, x0 + frac * (x1 - x0))
    if str(cfg.kind) == "img_exit":
        native = torch.flip(native, dims=(0,))
    return native.to(dtype=torch.float32)


def _conservation_eval(cfg: Any, model: torch.nn.Module, benchmark: Any, *, split: str, batch_size: int, device: torch.device, n_points: int, steps_per_batch: int, max_batches: int | None, seed: int, label: str) -> tuple[torch.Tensor, Any, dict[str, float], list[dict[str, float]], torch.Tensor, torch.Tensor, str]:
    kind = str(cfg.kind)
    if kind == "img_immse":
        grid, curve, metrics, rows = evaluate_immse_curve(
            model,
            benchmark,
            split=split,
            batch_size=batch_size,
            device=device,
            snr_min=float(cfg.eval.snr_min),
            snr_max=float(cfg.eval.snr_max),
            n_snr_points=int(n_points),
            max_batches=max_batches,
            steps_per_batch=int(steps_per_batch),
            seed=seed,
            progress_label=label,
        )
    elif kind == "img_exit":
        grid, curve, metrics, rows = evaluate_exit_curve(
            model,
            benchmark,
            split=split,
            batch_size=batch_size,
            device=device,
            n_s_points=int(n_points),
            max_batches=max_batches,
            steps_per_batch=int(steps_per_batch),
            seed=seed,
            progress_label=label,
        )
    elif kind == "img_qsc":
        grid, curve, metrics, rows = evaluate_qsc_curve(
            model,
            benchmark,
            split=split,
            batch_size=batch_size,
            device=device,
            t_min=float(cfg.eval.t_min),
            t_max=float(cfg.eval.t_max),
            n_t_points=int(n_points),
            max_batches=max_batches,
            steps_per_batch=int(steps_per_batch),
            seed=seed,
            progress_label=label,
        )
    else:
        raise ValueError(f"Unsupported CIFAR suite kind={kind!r}; expected img_exit, img_qsc, or img_immse.")
    area_x, density, coord_name, _ = _curve_coordinate_and_density(cfg, grid, curve)
    return grid, curve, metrics, rows, area_x, density, coord_name


def save_conservation_plot(path: Path, *, cfg: Any, grid: torch.Tensor, density: torch.Tensor, area: float, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = torch.as_tensor(grid, dtype=torch.float64).cpu()
    y = torch.nan_to_num(torch.as_tensor(density, dtype=torch.float64).cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(x, y, linewidth=2.2, color="tab:blue", label="conservation density")
    ax.fill_between(x, torch.zeros_like(y), y, color="tab:blue", alpha=0.18)
    if str(cfg.kind) == "img_immse":
        ax.set_xscale("log")
        ax.set_xlabel("SNR")
        ax.set_ylabel("0.5 * MMSE")
    elif str(cfg.kind) == "img_exit":
        ax.set_xlabel("mask rate s")
        ax.set_ylabel("gEXIT gap")
    else:
        ax.set_xlabel("pass-through probability t")
        ax.set_ylabel("QSC gEXIT")
    ax.set_title(f"{title} | integral={area:.6f} nats/token")
    ax.text(
        0.02,
        0.96,
        f"Integral = {area:.6f} nats/token",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82, "edgecolor": "0.8"},
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_trace_grid(path: Path, *, rows: list[tuple[int, list[torch.Tensor]]], title: str) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    n_cols = min(len(frames) for _, frames in rows)
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(1.6 * n_cols, 1.6 * len(rows)), squeeze=False)
    for row_idx, (num_steps, frames) in enumerate(rows):
        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            ax.imshow(torch.as_tensor(frames[col_idx], dtype=torch.float32).clamp(0.0, 1.0).numpy())
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(str(num_steps))
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_generated_grid(path: Path, *, images: torch.Tensor, rows: int, cols: int, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = int(rows)
    cols = int(cols)
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive.")
    needed = rows * cols
    if images.ndim != 4:
        raise ValueError(f"Expected images shaped [N,H,W,C] or [N,C,H,W], got {tuple(images.shape)}.")
    if int(images.shape[0]) < needed:
        raise ValueError(f"Need at least {needed} images for a {rows}x{cols} grid; got {int(images.shape[0])}.")
    imgs = images[:needed].detach().cpu().clamp(0.0, 1.0)
    # Accept either channels-first [N,C,H,W] or channels-last [N,H,W,C].
    if int(imgs.shape[1]) in (1, 3):
        imgs = imgs.permute(0, 2, 3, 1).contiguous()
    elif int(imgs.shape[-1]) in (1, 3):
        pass
    else:
        raise ValueError(f"Expected images shaped [N,H,W,C] or [N,C,H,W] with C in {{1,3}}, got {tuple(images.shape)}.")
    fig, axes = plt.subplots(rows, cols, figsize=(1.6 * cols, 1.6 * rows), squeeze=False)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r][c]
            img = imgs[idx]
            ax.imshow(img.numpy())
            ax.set_xticks([])
            ax.set_yticks([])
            idx += 1
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_metric_plots(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    for metric, filename, ylabel in [
        ("fid", "fid_vs_steps.png", "FID"),
        ("inception_score_mean", "inception_vs_steps.png", "Inception Score"),
    ]:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        for schedule_name in ("uniform", "equal_info", "cosine", "derivative"):
            schedule_rows = sorted((row for row in rows if row["schedule"] == schedule_name), key=lambda row: int(row["num_steps"]))
            if not schedule_rows:
                continue
            ax.plot(
                [int(row["num_steps"]) for row in schedule_rows],
                [float(row[metric]) for row in schedule_rows],
                marker="o",
                linewidth=2.0,
                label=schedule_name,
            )
        ax.set_xscale("log")
        ax.set_xlabel("diffusion steps")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=200)
        plt.close(fig)


def _wandb_image(path: Path, *, caption: str):
    import wandb

    return wandb.Image(str(path), caption=caption)


def _wandb_table(rows: list[dict[str, Any]]):
    import wandb

    if not rows:
        return wandb.Table(columns=[])
    columns = list(rows[0].keys())
    table = wandb.Table(columns=columns)
    for row in rows:
        table.add_data(*[row.get(column) for column in columns])
    return table


def _schedule_rows(*, weight_variant: str, schedule_name: str, num_steps: int, schedule: torch.Tensor, coord_name: str) -> list[dict[str, Any]]:
    return [
        {
            "weight_variant": weight_variant,
            "schedule": schedule_name,
            "num_steps": int(num_steps),
            "index": int(index),
            coord_name: float(value),
        }
        for index, value in enumerate(torch.as_tensor(schedule, dtype=torch.float64).tolist())
    ]


class _Progress:
    def __init__(self, *, total: int, label: str, print_every: int, enable_bar: bool):
        self.total = int(total)
        self.label = str(label)
        self.print_every = max(0, int(print_every))
        self._last_print = 0
        self._bar = tqdm(total=self.total, desc=self.label, unit="img", dynamic_ncols=True) if enable_bar else None

    def update_to(self, done: int) -> None:
        done = int(done)
        if self._bar is not None:
            delta = max(0, done - int(self._bar.n))
            if delta:
                self._bar.update(delta)
        if self.print_every > 0:
            if done >= self.total or done - self._last_print >= self.print_every:
                self._last_print = done
                print(f"[suite] {self.label} {done}/{self.total}", flush=True)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None


def _is_oom_error(err: BaseException) -> bool:
    if isinstance(err, torch.cuda.OutOfMemoryError):
        return True
    message = str(err).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def _probe_max_batch_size(*, start: int, max_batch: int, try_batch: Any, label: str) -> int:
    start = int(start)
    max_batch = int(max_batch)
    if start <= 0 or max_batch <= 0:
        raise ValueError("start and max_batch must be positive.")
    if start > max_batch:
        start = max_batch
    best = 0
    candidate = start
    print(f"[suite] probing {label} batch size (start={start} max={max_batch})", flush=True)
    while candidate <= max_batch:
        try:
            try_batch(int(candidate))
            best = int(candidate)
            candidate *= 2
        except RuntimeError as err:
            if not _is_oom_error(err):
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            break
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            break
    if best <= 0:
        raise RuntimeError(f"Could not find any feasible batch size for {label} (start={start}).")
    lo = best
    hi = min(max_batch, candidate - 1)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        try:
            try_batch(int(mid))
            lo = mid
        except RuntimeError as err:
            if not _is_oom_error(err):
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            hi = mid - 1
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            hi = mid - 1
    print(f"[suite] probe result {label} batch_size={lo}", flush=True)
    return int(lo)


@torch.no_grad()
def _probe_max_generation_batch_size(
    *,
    model: torch.nn.Module,
    benchmark: Any,
    cfg: Any,
    temperature: float,
    top_k: int,
    device: torch.device,
    seed: int,
    start: int,
    max_batch: int,
) -> int:
    schedule = build_uniform_schedule(cfg, num_steps=25)

    def _try(bs: int) -> None:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
        _images, _frames = sample_with_schedule(
            model,
            benchmark,
            cfg,
            schedule=schedule,
            count=int(bs),
            batch_size=int(bs),
            temperature=float(temperature),
            top_k=int(top_k),
            device=device,
            generator=gen,
            trace=False,
        )

    return _probe_max_batch_size(start=int(start), max_batch=int(max_batch), try_batch=_try, label="generation")


@torch.no_grad()
def _probe_max_eval_batch_size(
    *,
    model: torch.nn.Module,
    benchmark: Any,
    cfg: Any,
    device: torch.device,
    seed: int,
    start: int,
    max_batch: int,
    split: str,
) -> int:
    kind = str(cfg.kind)
    base_model = unwrap_model(model)
    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(int(seed))

    def _try(bs: int) -> None:
        batch = next(
            iter(
                benchmark.iter_eval_batches(
                    split=str(split),
                    batch_size=int(bs),
                    device=device,
                    include_next_token=False,
                    max_batches=1,
                    shard_rank=0,
                    num_shards=1,
                )
            )
        )
        if kind == "img_exit":
            s = torch.full((batch.shape[0],), 0.5, dtype=torch.float32, device=device)
            masked, _mask, _forced = apply_sequence_masking(batch, s, mask_token_id=base_model.mask_token_id, generator=cpu_generator)
            _ = model(masked, timesteps=s)
            return
        if kind == "img_qsc":
            t = torch.full((batch.shape[0],), 0.5, dtype=torch.float32, device=device)
            kernel = log_ksc_kernel(batch, t, int(base_model.vocab_size))
            outputs = model.forward_outputs(batch, timesteps=t, channel_tilt=kernel) if hasattr(model, "forward_outputs") else None
            _ = outputs.logits if outputs is not None else model(batch, timesteps=t) + kernel
            return
        if kind == "img_immse":
            snr = torch.full((batch.shape[0], benchmark.context_length, 1), 1.0, dtype=torch.float32, device=device)
            noisy_shape = (batch.shape[0], benchmark.context_length, int(base_model.x_values.embedding_dim))
            noisy = torch.randn(noisy_shape, dtype=torch.float32, device=device)
            _ = model(noisy, snr)
            return
        raise ValueError(f"Unsupported kind for eval-batch probing: {kind!r}")

    return _probe_max_batch_size(start=int(start), max_batch=int(max_batch), try_batch=_try, label="eval")


def run_variant(
    *,
    variant: str,
    state_dict: dict[str, Any],
    cfg: Any,
    benchmark: Any,
    real_images: torch.Tensor,
    output_dir: Path,
    split: str,
    sample_count: int,
    batch_size: int,
    eval_batch_size: int,
    auto_batch: bool,
    auto_eval_batch: bool,
    auto_batch_max: int,
    conservation_points: int,
    conservation_steps_per_batch: int,
    step_budgets: list[int],
    schedules_to_run: list[str],
    max_batches: int | None,
    temperature: float,
    top_k: int,
    seed: int,
    device: torch.device,
    grid_rows: int,
    grid_cols: int,
    grid_steps: int,
    trace_count: int,
    trace_frames: int,
    trace_steps: int,
    progress_print_every: int,
    disable_progress_bar: bool,
    wb_run: Any | None = None,
    progress_log_every: int = 0,
) -> list[dict[str, Any]]:
    variant_dir = ensure_dir(output_dir / variant)
    model = _make_model(cfg, benchmark, state_dict, device)
    if bool(auto_batch) or bool(auto_eval_batch):
        if device.type != "cuda":
            raise ValueError("--auto-batch/--auto-eval-batch require a CUDA device.")
        if bool(auto_batch):
            batch_size = int(
                _probe_max_generation_batch_size(
                    model=model,
                    benchmark=benchmark,
                    cfg=cfg,
                    temperature=float(temperature),
                    top_k=int(top_k),
                    device=device,
                    seed=int(seed) + (0 if variant == "final" else 10_000),
                    start=int(batch_size),
                    max_batch=int(auto_batch_max),
                )
            )
        if bool(auto_eval_batch):
            eval_batch_size = int(
                _probe_max_eval_batch_size(
                    model=model,
                    benchmark=benchmark,
                    cfg=cfg,
                    device=device,
                    seed=int(seed) + (123 if variant == "final" else 10_123),
                    start=int(eval_batch_size),
                    max_batch=int(auto_batch_max),
                    split=str(split),
                )
            )
        print(f"[suite] variant={variant} tuned batch_size={batch_size} eval_batch_size={eval_batch_size}", flush=True)
    print(f"[suite] variant={variant} conservation", flush=True)
    grid, _curve, curve_metrics, curve_rows, area_x, density, coord_name = _conservation_eval(
        cfg,
        model,
        benchmark,
        split=split,
        batch_size=eval_batch_size,
        device=device,
        n_points=conservation_points,
        steps_per_batch=conservation_steps_per_batch,
        max_batches=max_batches,
        seed=seed + (0 if variant == "final" else 10_000),
        label=f"{variant}/conservation",
    )
    save_csv(variant_dir / "conservation_curve.csv", curve_rows)
    save_json(variant_dir / "conservation_metrics.json", {key: float(value) if isinstance(value, (int, float)) else value for key, value in curve_metrics.items()})
    plot_density = 0.5 * torch.as_tensor(_curve, dtype=torch.float64) if str(cfg.kind) == "img_immse" else density
    save_conservation_plot(
        variant_dir / "conservation_law.png",
        cfg=cfg,
        grid=grid,
        density=plot_density,
        area=float(curve_metrics["curve_integral_nats"]),
        title=f"{variant} conservation law ({split})",
    )
    if wb_run is not None:
        wb_run.log(
            {
                f"{variant}/conservation_law_nats": float(curve_metrics["curve_integral_nats"]),
                f"{variant}/conservation_integral_nats": float(curve_metrics["curve_integral_nats"]),
                f"{variant}/conservation_points": int(conservation_points),
                f"{variant}/conservation_law": _wandb_image(variant_dir / "conservation_law.png", caption=f"{variant} conservation law; integral={float(curve_metrics['curve_integral_nats']):.6f}"),
                "eval_suite/latest_conservation_law_nats": float(curve_metrics["curve_integral_nats"]),
            }
        )

    schedule_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    trace_rows: dict[str, list[tuple[int, list[torch.Tensor]]]] = {"uniform": [], "equal_info": [], "cosine": [], "derivative": []}
    start_time = time.perf_counter()
    for num_steps in step_budgets:
        schedules = {
            "uniform": build_uniform_schedule(cfg, num_steps=int(num_steps)),
            "equal_info": build_equal_info_schedule(cfg, area_x=area_x, density=density, num_steps=int(num_steps)),
            "cosine": build_cosine_schedule(cfg, num_steps=int(num_steps)),
            "derivative": build_derivative_schedule(cfg, area_x=area_x, density=density, num_steps=int(num_steps)),
        }
        for schedule_name in schedules_to_run:
            schedule = schedules[schedule_name]
            schedule_rows.extend(_schedule_rows(weight_variant=variant, schedule_name=schedule_name, num_steps=int(num_steps), schedule=schedule, coord_name=coord_name))
            generator = torch.Generator(device=device)
            generator.manual_seed(_variant_seed(seed, variant=variant, num_steps=int(num_steps)))
            print(f"[suite] variant={variant} schedule={schedule_name} steps={num_steps} samples={sample_count}", flush=True)
            next_progress_log = max(1, int(progress_log_every)) if int(progress_log_every) > 0 else None
            enable_bar = (not bool(disable_progress_bar)) and bool(sys.stdout.isatty())
            progress = _Progress(
                total=int(sample_count),
                label=f"{variant}/{schedule_name}/{int(num_steps)}",
                print_every=int(progress_print_every),
                enable_bar=enable_bar,
            )

            def _progress_callback(done: int) -> None:
                nonlocal next_progress_log
                progress.update_to(int(done))
                if wb_run is None or next_progress_log is None:
                    return
                while int(done) >= int(next_progress_log):
                    wb_run.log(
                        {
                            f"{variant}/{schedule_name}/generated_samples": int(next_progress_log),
                            f"{variant}/{schedule_name}/generation_progress_fraction": float(next_progress_log) / float(sample_count),
                            f"{variant}/{schedule_name}/num_steps": int(num_steps),
                            "eval_suite/latest_generated_samples": int(next_progress_log),
                        }
                    )
                    next_progress_log += int(progress_log_every)

            generated, _ = sample_with_schedule(
                model,
                benchmark,
                cfg,
                schedule=schedule,
                count=int(sample_count),
                batch_size=batch_size,
                temperature=temperature,
                top_k=top_k,
                device=device,
                generator=generator,
                trace=False,
                progress_callback=_progress_callback,
            )
            progress.update_to(int(sample_count))
            progress.close()
            if int(grid_steps) > 0 and int(num_steps) == int(grid_steps):
                save_generated_grid(
                    variant_dir / f"generated_grid_{schedule_name}_steps_{int(num_steps)}.png",
                    images=generated,
                    rows=int(grid_rows),
                    cols=int(grid_cols),
                    title=f"{variant}: {schedule_name} generated samples ({int(num_steps)} steps)",
                )
            metrics = compute_generation_metrics(
                generated,
                real_images,
                sample_count=int(sample_count),
                batch_size=batch_size,
                device=device,
            )
            trace_target_steps = int(trace_steps)
            want_traces = int(trace_count) > 0 and (trace_target_steps == 0 or int(num_steps) == trace_target_steps)
            if want_traces:
                for trace_idx in range(int(trace_count)):
                    trace_generator = torch.Generator(device=device)
                    trace_generator.manual_seed(_variant_seed(seed + 10_000 * trace_idx, variant=variant, num_steps=int(num_steps)))
                    _, trace_frames_out = sample_with_schedule(
                        model,
                        benchmark,
                        cfg,
                        schedule=schedule,
                        count=1,
                        batch_size=1,
                        temperature=temperature,
                        top_k=top_k,
                        device=device,
                        generator=trace_generator,
                        trace=True,
                        trace_frame_count=int(trace_frames),
                    )
                    trace_rows[schedule_name].append((int(num_steps), trace_frames_out))
            row = {
                "weight_variant": variant,
                "schedule": schedule_name,
                "num_steps": int(num_steps),
                "sample_count": int(sample_count),
                "split": str(split),
                "fid": float(metrics["fid"]),
                "inception_score_mean": float(metrics["inception_score_mean"]),
                "inception_score_std": float(metrics["inception_score_std"]),
                "real_images_repeated": bool(metrics["real_images_repeated"]),
                "conservation_integral": float(curve_metrics["curve_integral_nats"]),
                "elapsed_seconds": float(time.perf_counter() - start_time),
            }
            metric_rows.append(row)
            save_csv(variant_dir / "generation_metrics.csv", metric_rows)
            save_json(variant_dir / "generation_metrics.json", {"weight_variant": variant, "rows": metric_rows})
            if wb_run is not None:
                wb_run.log(
                    {
                        f"{variant}/{schedule_name}/fid": float(row["fid"]),
                        f"{variant}/{schedule_name}/inception_score_mean": float(row["inception_score_mean"]),
                        f"{variant}/{schedule_name}/inception_score_std": float(row["inception_score_std"]),
                        f"{variant}/{schedule_name}/num_steps": int(num_steps),
                        "eval_suite/latest_fid": float(row["fid"]),
                        "eval_suite/latest_inception_score_mean": float(row["inception_score_mean"]),
                    }
                )
    save_csv(variant_dir / "schedules.csv", schedule_rows)
    if trace_rows["uniform"]:
        save_trace_grid(variant_dir / "generation_traces_uniform.png", rows=trace_rows["uniform"], title=f"{variant}: uniform schedule traces")
    if trace_rows["equal_info"]:
        save_trace_grid(variant_dir / "generation_traces_equal_info.png", rows=trace_rows["equal_info"], title=f"{variant}: equal-info schedule traces")
    if trace_rows["cosine"]:
        save_trace_grid(variant_dir / "generation_traces_cosine.png", rows=trace_rows["cosine"], title=f"{variant}: cosine schedule traces")
    if trace_rows["derivative"]:
        save_trace_grid(variant_dir / "generation_traces_derivative.png", rows=trace_rows["derivative"], title=f"{variant}: derivative schedule traces")
    save_metric_plots(variant_dir, metric_rows)
    if wb_run is not None:
        payload = {
            f"{variant}/conservation_integral": float(curve_metrics["curve_integral_nats"]),
            f"{variant}/conservation_curve": _wandb_table(curve_rows),
            f"{variant}/generation_metrics": _wandb_table(metric_rows),
            f"{variant}/schedules": _wandb_table(schedule_rows),
            f"{variant}/conservation_law": _wandb_image(variant_dir / "conservation_law.png", caption=f"{variant} conservation law"),
            f"{variant}/fid_vs_steps": _wandb_image(variant_dir / "fid_vs_steps.png", caption=f"{variant} FID vs steps"),
            f"{variant}/inception_vs_steps": _wandb_image(variant_dir / "inception_vs_steps.png", caption=f"{variant} Inception vs steps"),
        }
        if (variant_dir / "generation_traces_uniform.png").is_file():
            payload[f"{variant}/generation_traces_uniform"] = _wandb_image(variant_dir / "generation_traces_uniform.png", caption=f"{variant} uniform traces")
        if (variant_dir / "generation_traces_equal_info.png").is_file():
            payload[f"{variant}/generation_traces_equal_info"] = _wandb_image(variant_dir / "generation_traces_equal_info.png", caption=f"{variant} equal-info traces")
        if (variant_dir / "generation_traces_cosine.png").is_file():
            payload[f"{variant}/generation_traces_cosine"] = _wandb_image(variant_dir / "generation_traces_cosine.png", caption=f"{variant} cosine traces")
        if (variant_dir / "generation_traces_derivative.png").is_file():
            payload[f"{variant}/generation_traces_derivative"] = _wandb_image(variant_dir / "generation_traces_derivative.png", caption=f"{variant} derivative traces")
        wb_run.log(payload)
    save_json(
        variant_dir / "manifest.json",
        {
            "weight_variant": variant,
            "kind": str(cfg.kind),
            "split": str(split),
            "sample_count": int(sample_count),
            "conservation_points": int(conservation_points),
            "conservation_steps_per_batch": int(conservation_steps_per_batch),
            "step_budgets": [int(value) for value in step_budgets],
            "schedules": list(schedules_to_run),
            "outputs": {
                "conservation_curve": "conservation_curve.csv",
                "conservation_law": "conservation_law.png",
                "schedules": "schedules.csv",
                "generation_metrics": "generation_metrics.csv",
                "generation_traces_uniform": "generation_traces_uniform.png",
                "generation_traces_equal_info": "generation_traces_equal_info.png",
                "generation_traces_cosine": "generation_traces_cosine.png",
                "generation_traces_derivative": "generation_traces_derivative.png",
                "fid_vs_steps": "fid_vs_steps.png",
                "inception_vs_steps": "inception_vs_steps.png",
            },
        },
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metric_rows


def main(argv: list[str] | None = None) -> int:
    configure_environment()
    args = build_arg_parser().parse_args(argv)
    set_global_seed(int(args.seed), deterministic=False)
    device = get_best_device(str(args.device))
    cache_dir = ensure_dir(Path(args.cache_dir).expanduser() if args.cache_dir is not None else _default_cache_dir(args.run))
    output_dir = ensure_dir(Path(args.output_dir).expanduser() if args.output_dir is not None else _default_output_dir(args.run))
    config_path, final_checkpoint = ensure_cached_config_and_final_checkpoint(run=args.run, cache_dir=cache_dir, entity=args.entity, project=args.project)
    cfg = OmegaConf.load(config_path)
    model_cfg = _cfg_unwrap(getattr(cfg, "model", None))
    if isinstance(model_cfg, dict):
        cfg.model = OmegaConf.create(model_cfg)
    eval_cfg = _cfg_unwrap(getattr(cfg, "eval", None))
    if isinstance(eval_cfg, dict):
        cfg.eval = OmegaConf.create(eval_cfg)
    cfg_kind = _cfg_string(getattr(cfg, "kind", None))
    if cfg_kind not in {"img_exit", "img_qsc", "img_immse"}:
        raise ValueError(f"Unsupported CIFAR suite kind={cfg_kind!r}; expected img_exit, img_qsc, or img_immse.")
    # Normalize any wrapped Hydra scalar so downstream helpers see a plain string.
    cfg.kind = cfg_kind
    benchmark_cfg = _cfg_unwrap(getattr(cfg, "benchmark", None))
    if isinstance(benchmark_cfg, dict):
        benchmark_cfg = OmegaConf.create(benchmark_cfg)
    # Ensure downstream helpers that read cfg.benchmark.* see a structured config, not a wrapped dict.
    cfg.benchmark = benchmark_cfg
    benchmark = load_img_benchmark(benchmark_cfg, project_root=find_repo_root(APP_ROOT))
    real_images = benchmark.split_images(str(args.split))
    if real_images is None:
        raise ValueError(f"Benchmark split {args.split!r} does not include real images for FID.")
    step_budgets = _parse_step_budgets(str(args.step_budgets))
    weight_variants = ["ema"] if bool(args.ema_only) else _parse_choices(str(args.weight_variants), allowed={"final", "ema"}, arg_name="--weight-variants")
    schedules_to_run = _parse_choices(str(args.schedules), allowed={"uniform", "equal_info", "cosine", "derivative"}, arg_name="--schedules")
    states = load_final_and_ema_state_dicts(final_checkpoint, required_variants=weight_variants)
    batch_size = int(args.batch_size if args.batch_size is not None else getattr(cfg.eval, "batch_size", 64))
    eval_batch_size = int(args.eval_batch_size if args.eval_batch_size is not None else getattr(cfg.eval, "batch_size", batch_size))
    if (
        int(args.sample_count) <= 0
        or batch_size <= 0
        or eval_batch_size <= 0
        or int(args.conservation_points) <= 1
        or int(args.conservation_steps_per_batch) <= 0
        or int(args.grid_rows) <= 0
        or int(args.grid_cols) <= 0
        or int(args.trace_count) < 0
        or int(args.trace_frames) <= 1
    ):
        raise ValueError("sample-count, batch-size, eval-batch-size, conservation-steps-per-batch must be positive and conservation-points must be > 1.")
    if int(args.grid_steps) > 0 and int(args.grid_rows) * int(args.grid_cols) > int(args.sample_count):
        raise ValueError("grid-rows * grid-cols must be <= sample-count for the selected grid-steps.")

    wb_run = None
    if bool(args.log_wandb):
        import wandb

        entity_value = args.entity or getattr(cfg.logging, "entity", "") or None
        entity = str(entity_value) if entity_value else None
        project = str(args.project or getattr(cfg.logging, "project", "") or "gexit-diffusion")
        source_group = str(getattr(cfg.logging, "group", "cifar"))
        group = str(args.wandb_group or f"{source_group}_eval_suite")
        run_name = str(args.wandb_name or f"cifar_suite_{_safe_run_id(str(args.run))}_step_{checkpoint_step(final_checkpoint):07d}")
        wb_run = wandb.init(
            entity=entity,
            project=project,
            group=group,
            name=run_name,
            job_type="eval_cifar_suite",
            config={
                "source_run": str(args.run),
                "source_checkpoint": str(final_checkpoint),
                "source_checkpoint_step": int(checkpoint_step(final_checkpoint)),
                "kind": str(cfg.kind),
                "sample_count": int(args.sample_count),
                "split": str(args.split),
                "conservation_points": int(args.conservation_points),
                "conservation_steps_per_batch": int(args.conservation_steps_per_batch),
                "step_budgets": step_budgets,
                "weight_variants": weight_variants,
                "schedules": schedules_to_run,
                "output_dir": str(output_dir),
            },
        )

    all_rows: list[dict[str, Any]] = []
    for variant in weight_variants:
        all_rows.extend(
            run_variant(
                variant=variant,
                state_dict=states[variant],
                cfg=cfg,
                benchmark=benchmark,
                real_images=real_images,
                output_dir=output_dir,
                split=str(args.split),
                sample_count=int(args.sample_count),
                batch_size=batch_size,
                eval_batch_size=eval_batch_size,
                auto_batch=bool(args.auto_batch),
                auto_eval_batch=bool(args.auto_eval_batch),
                auto_batch_max=int(args.auto_batch_max),
                conservation_points=int(args.conservation_points),
                conservation_steps_per_batch=int(args.conservation_steps_per_batch),
                step_budgets=step_budgets,
                schedules_to_run=schedules_to_run,
                max_batches=None if args.max_batches is None else int(args.max_batches),
                temperature=float(args.temperature),
                top_k=int(args.top_k),
                seed=int(args.seed),
                device=device,
                grid_rows=int(args.grid_rows),
                grid_cols=int(args.grid_cols),
                grid_steps=int(args.grid_steps),
                trace_count=int(args.trace_count),
                trace_frames=int(args.trace_frames),
                trace_steps=int(args.trace_steps),
                progress_print_every=int(args.progress_print_every),
                disable_progress_bar=bool(args.no_progress_bar),
                wb_run=wb_run,
                progress_log_every=int(args.progress_log_every),
            )
        )

    summary_rows = [
        {
            "weight_variant": row["weight_variant"],
            "schedule": row["schedule"],
            "num_steps": row["num_steps"],
            "fid": row["fid"],
            "inception_score_mean": row["inception_score_mean"],
            "inception_score_std": row["inception_score_std"],
            "conservation_integral": row["conservation_integral"],
        }
        for row in all_rows
    ]
    save_csv(output_dir / "summary_metrics.csv", summary_rows)
    save_json(
        output_dir / "suite_manifest.json",
        {
            "run": str(args.run),
            "config": str(config_path),
            "checkpoint": str(final_checkpoint),
            "checkpoint_step": int(checkpoint_step(final_checkpoint)),
            "cache_dir": str(cache_dir),
            "output_dir": str(output_dir),
            "kind": str(cfg.kind),
            "benchmark": benchmark.benchmark_metadata(),
            "device": str(device),
            "sample_count": int(args.sample_count),
            "split": str(args.split),
            "conservation_points": int(args.conservation_points),
            "conservation_steps_per_batch": int(args.conservation_steps_per_batch),
            "step_budgets": step_budgets,
            "variants": weight_variants,
            "schedules": schedules_to_run,
        },
    )
    if wb_run is not None:
        wb_run.log(
            {
                "eval_suite/summary_metrics": _wandb_table(summary_rows),
            }
        )
        artifact = wandb.Artifact(f"{_safe_run_id(str(args.run))}-cifar-suite-step-{checkpoint_step(final_checkpoint):07d}", type="eval-suite")
        artifact.add_file(str(output_dir / "summary_metrics.csv"))
        artifact.add_file(str(output_dir / "suite_manifest.json"))
        for variant in weight_variants:
            variant_dir = output_dir / variant
            for filename in (
                "conservation_curve.csv",
                "conservation_metrics.json",
                "generation_metrics.csv",
                "generation_metrics.json",
                "schedules.csv",
                "manifest.json",
                "conservation_law.png",
                "generation_traces_uniform.png",
                "generation_traces_equal_info.png",
                "generation_traces_cosine.png",
                "fid_vs_steps.png",
                "inception_vs_steps.png",
            ):
                path = variant_dir / filename
                if path.is_file():
                    artifact.add_file(str(path), name=f"{variant}/{filename}")
        wb_run.log_artifact(artifact)
        wb_run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
