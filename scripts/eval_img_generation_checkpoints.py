from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from experiments.common.io import ensure_dir, save_csv, save_json
from experiments.common.repro import configure_environment, get_best_device, set_global_seed
from scripts.img_common import find_repo_root, load_img_benchmark
from src.channels import apply_ksc_noise, log_ksc_kernel
from src.exit import apply_sequence_masking
from src.img_data import save_image_grid
from src.img_models import RGB8ContinuousDDPMModel, RGB8DistributionDDPMModel, RGB8ImageTokenUNet
from src.lm_distributed import unwrap_model
from src.models import posterior_mean_from_logits


CHECKPOINT_RE = re.compile(r"model_step_(\d+)\.pt$")
DEFAULT_WANDB_ENTITY: str | None = None
DEFAULT_WANDB_PROJECT = "gexit-diffusion"


def unwrap_wandb_config_values(cfg: Any) -> Any:
    """Convert a W&B-downloaded config.yaml with top-level `{value: ...}` entries."""
    container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(container, dict):
        return cfg
    if not any(isinstance(value, dict) and "value" in value for value in container.values()):
        return cfg
    unwrapped = {
        key: value["value"] if isinstance(value, dict) and "value" in value else value
        for key, value in container.items()
        if key != "_wandb"
    }
    return OmegaConf.create(unwrapped)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate image generation FID/IS over all checkpoints in a W&B run.")
    parser.add_argument("--run", required=True, help="W&B run path entity/project/run_id, or a bare run id.")
    parser.add_argument("--entity", default=None, help="W&B entity for bare run ids. Defaults to config logging.entity.")
    parser.add_argument("--project", default=None, help="W&B project for bare run ids. Defaults to config logging.project.")
    parser.add_argument("--cache-dir", default=None, help="Directory containing or receiving downloaded config/checkpoints.")
    parser.add_argument("--output-dir", default=None, help="Directory for eval metrics and sample grids.")
    parser.add_argument("--sample-count", type=int, default=10_000, help="Generated image count per checkpoint.")
    parser.add_argument("--num-diff-steps", type=int, default=64, help="Reverse corruption steps per generated batch.")
    parser.add_argument("--batch-size", type=int, default=64, help="Generation and metric batch size.")
    parser.add_argument("--split", default="test", help="Real-image split for FID.")
    parser.add_argument("--seed", type=int, default=2026, help="Deterministic sampling seed.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for discrete posterior logits.")
    parser.add_argument("--top-k", type=int, default=0, help="Optional top-k truncation for posterior sampling; 0 disables it.")
    parser.add_argument("--hold-fraction", type=float, default=0.0, help="For QSC sampling: progressively freeze this fraction of tokens (EXIT-like).")
    parser.add_argument("--device", default="auto", help="Torch device, or auto.")
    parser.add_argument("--log-wandb", action="store_true", help="Log per-checkpoint eval metrics to the source W&B run.")
    parser.add_argument("--save-generated", action="store_true", help="Save generated image tensors for each checkpoint.")
    parser.add_argument("--max-checkpoints", type=int, default=None, help="Optional limit for quick/debug evaluation.")
    parser.add_argument("--skip-metrics", action="store_true", help="Generate samples and write grids without FID/IS computation.")
    return parser


def _safe_run_id(run: str) -> str:
    return str(run).rstrip("/").split("/")[-1]


def _default_cache_dir(run: str) -> Path:
    return APP_ROOT / "results" / "img" / "generative_eval" / _safe_run_id(run) / "downloads"


def _default_output_dir(run: str) -> Path:
    return APP_ROOT / "results" / "img" / "generative_eval" / _safe_run_id(run)


def checkpoint_step(path: str | Path) -> int:
    match = CHECKPOINT_RE.search(Path(path).name)
    if match is None:
        raise ValueError(f"Could not parse checkpoint step from {path}.")
    return int(match.group(1))


def discover_checkpoints(cache_dir: str | Path, *, max_checkpoints: int | None = None) -> list[Path]:
    root = Path(cache_dir)
    checkpoints = sorted(root.glob("checkpoints/model_step_*.pt"), key=checkpoint_step)
    if max_checkpoints is not None:
        checkpoints = checkpoints[: int(max_checkpoints)]
    return checkpoints


def _resolve_run_path(run: str, *, cfg: Any | None, entity: str | None, project: str | None) -> str:
    parts = [part for part in str(run).split("/") if part]
    if len(parts) == 3:
        return "/".join(parts)
    if len(parts) != 1:
        raise ValueError("--run must be either entity/project/run_id or a bare run id.")
    resolved_entity = entity or (None if cfg is None else str(getattr(cfg.logging, "entity", "") or "")) or DEFAULT_WANDB_ENTITY
    resolved_project = project or (None if cfg is None else str(getattr(cfg.logging, "project", "") or "")) or DEFAULT_WANDB_PROJECT
    if not resolved_entity or not resolved_project:
        raise ValueError("Bare run ids require --entity and --project when config.yaml is not already cached.")
    return f"{resolved_entity}/{resolved_project}/{parts[0]}"


def _download_wandb_run_files(*, run: str, cache_dir: Path, entity: str | None, project: str | None) -> Any:
    import wandb

    api = wandb.Api()
    cached_cfg = OmegaConf.load(cache_dir / "config.yaml") if (cache_dir / "config.yaml").is_file() else None
    run_path = _resolve_run_path(run, cfg=cached_cfg, entity=entity, project=project)
    wb_run = api.run(run_path)
    for file_obj in wb_run.files():
        name = str(file_obj.name)
        if name == "config.yaml" or name.startswith("checkpoints/model_step_"):
            target = cache_dir / name
            if target.is_file():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            file_obj.download(root=str(cache_dir), replace=False)
    return wb_run


def ensure_cached_run_files(*, run: str, cache_dir: Path, entity: str | None, project: str | None, max_checkpoints: int | None) -> tuple[Path, list[Path], Any | None]:
    config_path = cache_dir / "config.yaml"
    checkpoints = discover_checkpoints(cache_dir, max_checkpoints=max_checkpoints)
    if config_path.is_file() and checkpoints:
        return config_path, checkpoints, None
    wb_run = _download_wandb_run_files(run=run, cache_dir=cache_dir, entity=entity, project=project)
    checkpoints = discover_checkpoints(cache_dir, max_checkpoints=max_checkpoints)
    if not config_path.is_file():
        raise FileNotFoundError(f"W&B run did not provide config.yaml under {cache_dir}.")
    if not checkpoints:
        raise FileNotFoundError(f"W&B run did not provide checkpoints/model_step_*.pt under {cache_dir}.")
    return config_path, checkpoints, wb_run


def _normalize_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {key[len("module."):] if key.startswith("module.") else key: value for key, value in state_dict.items()}


def load_model_state_dict(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu")
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict"):
            candidate = payload.get(key)
            if isinstance(candidate, dict):
                return _normalize_state_dict_keys(candidate)
        if all(isinstance(key, str) for key in payload):
            return _normalize_state_dict_keys(payload)
    raise TypeError(f"Could not extract model state dict from {path}.")


def build_image_model(cfg: Any, *, vocab_size: int, context_length: int) -> torch.nn.Module:
    kind = str(cfg.kind)
    name = str(cfg.model.name).lower()
    if kind in {"img_exit", "img_qsc"}:
        if name == "img_unet_rgb8":
            return RGB8ImageTokenUNet(
                vocab_size=vocab_size,
                image_size=int(cfg.benchmark.image_size),
                base_channels=int(cfg.model.base_channels),
                channel_multipliers=tuple(int(value) for value in cfg.model.channel_multipliers),
                dropout=float(cfg.model.dropout),
                time_embed_input_dim=int(cfg.model.time_embed_input_dim),
                time_embed_dim=int(cfg.model.time_embed_dim),
                input_skip=bool(cfg.model.get("input_skip", False)),
            )
        raise ValueError(f"Image generation eval only supports img_unet_rgb8 for kind={kind!r}; got model.name={cfg.model.name!r}.")
    if kind == "img_immse":
        if name == "img_unet_rgb8_continuous":
            return RGB8ContinuousDDPMModel(
                vocab_size=vocab_size,
                image_size=int(cfg.benchmark.image_size),
                base_channels=int(cfg.model.base_channels),
                channel_multipliers=tuple(int(value) for value in cfg.model.channel_multipliers),
                dropout=float(cfg.model.dropout),
                time_embed_input_dim=int(cfg.model.time_embed_input_dim),
                time_embed_dim=int(cfg.model.time_embed_dim),
                input_skip=bool(cfg.model.get("input_skip", False)),
            )
        if name == "img_unet_rgb8_distribution":
            return RGB8DistributionDDPMModel(
                vocab_size=vocab_size,
                image_size=int(cfg.benchmark.image_size),
                base_channels=int(cfg.model.base_channels),
                channel_multipliers=tuple(int(value) for value in cfg.model.channel_multipliers),
                dropout=float(cfg.model.dropout),
                time_embed_input_dim=int(cfg.model.time_embed_input_dim),
                time_embed_dim=int(cfg.model.time_embed_dim),
            )
        raise ValueError(f"Image generation eval only supports img_unet_rgb8_* for I-MMSE; got model.name={cfg.model.name!r}.")
    raise ValueError(f"Unsupported image generation eval kind={kind!r}.")


def sample_from_logits(logits: torch.Tensor, *, temperature: float, top_k: int, generator: torch.Generator | None) -> torch.Tensor:
    if float(temperature) <= 0.0:
        return torch.argmax(logits, dim=-1)
    scaled = logits / float(temperature)
    if int(top_k) > 0 and int(top_k) < int(scaled.shape[-1]):
        values, _indices = torch.topk(scaled, k=int(top_k), dim=-1)
        cutoff = values[..., -1:].expand_as(scaled)
        scaled = torch.where(scaled >= cutoff, scaled, torch.full_like(scaled, -torch.inf))
    probs = torch.softmax(scaled, dim=-1)
    flat_probs = probs.reshape(-1, probs.shape[-1])
    if generator is None or torch.device(generator.device).type == probs.device.type:
        samples = torch.multinomial(flat_probs, num_samples=1, generator=generator)
    else:
        samples = torch.multinomial(flat_probs.cpu(), num_samples=1, generator=generator).to(device=probs.device)
    return samples.view(*probs.shape[:-1])


@torch.no_grad()
def sample_exit_images(
    model: torch.nn.Module,
    benchmark: Any,
    *,
    count: int,
    batch_size: int,
    num_diff_steps: int,
    temperature: float,
    top_k: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    base_model = unwrap_model(model)
    generated: list[torch.Tensor] = []
    schedule = torch.linspace(1.0, 0.0, steps=int(num_diff_steps) + 1, dtype=torch.float32)
    while sum(int(chunk.shape[0]) for chunk in generated) < int(count):
        current_batch = min(int(batch_size), int(count) - sum(int(chunk.shape[0]) for chunk in generated))
        clean = torch.full((current_batch, benchmark.context_length), int(base_model.mask_token_id), dtype=torch.long, device=device)
        observed = clean
        for step_idx in range(int(num_diff_steps)):
            s = schedule[step_idx].expand(current_batch).to(device=device)
            next_s = schedule[step_idx + 1].expand(current_batch).to(device=device)
            logits = model(observed, timesteps=s)
            clean = sample_from_logits(logits, temperature=temperature, top_k=top_k, generator=generator)
            observed, _, _ = apply_sequence_masking(clean, next_s, mask_token_id=int(base_model.mask_token_id), generator=generator)
        generated.append(benchmark.decode_token_sequences(clean.cpu()))
    return torch.cat(generated, dim=0)[: int(count)].clamp(0.0, 1.0)


@torch.no_grad()
def sample_qsc_images(
    model: torch.nn.Module,
    benchmark: Any,
    *,
    cfg: Any,
    count: int,
    batch_size: int,
    num_diff_steps: int,
    temperature: float,
    top_k: int,
    hold_fraction: float = 0.0,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    base_model = unwrap_model(model)
    generated: list[torch.Tensor] = []
    t_min = float(getattr(cfg.eval, "t_min", getattr(cfg.qsc, "train_t_min", 1.0 / float(base_model.vocab_size))))
    t_max = float(getattr(cfg.eval, "t_max", getattr(cfg.qsc, "train_t_max", 0.9999)))
    schedule = torch.linspace(t_min, t_max, steps=int(num_diff_steps) + 1, dtype=torch.float32)
    while sum(int(chunk.shape[0]) for chunk in generated) < int(count):
        current_batch = min(int(batch_size), int(count) - sum(int(chunk.shape[0]) for chunk in generated))
        clean = torch.randint(int(base_model.vocab_size), (current_batch, benchmark.context_length), generator=generator, dtype=torch.long).to(device=device)
        observed = clean
        frozen = None
        if float(hold_fraction) > 0.0:
            if not 0.0 <= float(hold_fraction) <= 1.0:
                raise ValueError(f"hold_fraction must be in [0, 1], got {hold_fraction}.")
            frozen = torch.zeros_like(clean, dtype=torch.bool)
        for step_idx in range(int(num_diff_steps)):
            t = schedule[step_idx].expand(current_batch).to(device=device)
            next_t = schedule[step_idx + 1].expand(current_batch).to(device=device)
            kernel = log_ksc_kernel(observed, t, int(base_model.vocab_size))
            outputs = model.forward_outputs(observed, timesteps=t, channel_tilt=kernel) if hasattr(model, "forward_outputs") else None
            logits = outputs.logits if outputs is not None else model(observed, timesteps=t) + kernel
            proposed = sample_from_logits(logits, temperature=temperature, top_k=top_k, generator=generator)
            if frozen is not None:
                clean = torch.where(frozen, clean, proposed)
                # Grow the frozen mask over time, similar to progressively unmasking in EXIT.
                target_frac = float(hold_fraction) * float(step_idx + 1) / float(max(1, int(num_diff_steps)))
                if target_frac > 0.0:
                    target_count = int(round(target_frac * float(clean.shape[1])))
                    if target_count > 0:
                        # Sample additional positions to freeze uniformly from the remaining unfrozen indices.
                        remaining = (~frozen).to(dtype=torch.float32)
                        remaining_counts = remaining.sum(dim=1)
                        for b in range(int(clean.shape[0])):
                            need = target_count - int(frozen[b].sum().item())
                            if need <= 0:
                                continue
                            if remaining_counts[b].item() <= 0:
                                continue
                            weights = remaining[b]
                            idx = torch.multinomial(weights, num_samples=min(int(need), int(remaining_counts[b].item())), replacement=False, generator=generator)
                            frozen[b, idx] = True
                # Re-noise only the unfrozen positions; frozen positions stay as-is.
                noisy, _ = apply_ksc_noise(clean, next_t, int(base_model.vocab_size), generator=generator)
                observed = torch.where(frozen, clean, noisy)
            else:
                clean = proposed
                observed, _ = apply_ksc_noise(clean, next_t, int(base_model.vocab_size), generator=generator)
        generated.append(benchmark.decode_token_sequences(clean.cpu()))
    return torch.cat(generated, dim=0)[: int(count)].clamp(0.0, 1.0)


@torch.no_grad()
def sample_qsc_images_keep_by_t(
    model: torch.nn.Module,
    benchmark: Any,
    *,
    cfg: Any,
    count: int,
    batch_size: int,
    num_diff_steps: int,
    temperature: float,
    top_k: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    """QSC sampler with a monotone keep-mask driven by the QSC pass-through events.

    This mimics the masked sampler behavior: once a position is revealed (passes through),
    it is kept fixed for the remainder of the generation chain.
    """
    base_model = unwrap_model(model)
    generated: list[torch.Tensor] = []
    t_min = float(getattr(cfg.eval, "t_min", getattr(cfg.qsc, "train_t_min", 1.0 / float(base_model.vocab_size))))
    t_max = float(getattr(cfg.eval, "t_max", getattr(cfg.qsc, "train_t_max", 0.9999)))
    schedule = torch.linspace(t_min, t_max, steps=int(num_diff_steps) + 1, dtype=torch.float32)

    while sum(int(chunk.shape[0]) for chunk in generated) < int(count):
        current_batch = min(int(batch_size), int(count) - sum(int(chunk.shape[0]) for chunk in generated))
        clean = torch.randint(int(base_model.vocab_size), (current_batch, benchmark.context_length), generator=generator, dtype=torch.long).to(device=device)
        observed = clean
        keep = torch.zeros_like(clean, dtype=torch.bool)

        for step_idx in range(int(num_diff_steps)):
            t = schedule[step_idx].expand(current_batch).to(device=device)
            next_t = schedule[step_idx + 1].expand(current_batch).to(device=device)
            kernel = log_ksc_kernel(observed, t, int(base_model.vocab_size))
            outputs = model.forward_outputs(observed, timesteps=t, channel_tilt=kernel) if hasattr(model, "forward_outputs") else None
            logits = outputs.logits if outputs is not None else model(observed, timesteps=t) + kernel
            proposed = sample_from_logits(logits, temperature=temperature, top_k=top_k, generator=generator)
            clean = torch.where(keep, clean, proposed)

            noisy, pass_mask = apply_ksc_noise(clean, next_t, int(base_model.vocab_size), generator=generator)
            keep = keep | pass_mask
            observed = torch.where(keep, clean, noisy)

        generated.append(benchmark.decode_token_sequences(clean.cpu()))
    return torch.cat(generated, dim=0)[: int(count)].clamp(0.0, 1.0)


@torch.no_grad()
def sample_immse_images(
    model: torch.nn.Module,
    benchmark: Any,
    *,
    cfg: Any,
    count: int,
    batch_size: int,
    num_diff_steps: int,
    temperature: float,
    top_k: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    base_model = unwrap_model(model)
    generated: list[torch.Tensor] = []
    snr_min = float(getattr(cfg.eval, "snr_min", getattr(cfg.train, "snr_min", 1.0e-6)))
    snr_max = float(getattr(cfg.eval, "snr_max", getattr(cfg.train, "snr_max", 1.0)))
    schedule = torch.exp(torch.linspace(math.log(snr_min), math.log(snr_max), steps=int(num_diff_steps) + 1, dtype=torch.float32))
    while sum(int(chunk.shape[0]) for chunk in generated) < int(count):
        current_batch = min(int(batch_size), int(count) - sum(int(chunk.shape[0]) for chunk in generated))
        snr0 = schedule[0].expand(current_batch, benchmark.context_length, 1).to(device=device)
        noisy = torch.randn((current_batch, benchmark.context_length, int(base_model.x_values.embedding_dim)), generator=generator, dtype=torch.float32).to(device=device)
        noisy = noisy / torch.sqrt(torch.clamp(snr0, min=1.0e-12))
        tokens = None
        for step_idx in range(int(num_diff_steps)):
            snr = schedule[step_idx].expand(current_batch, benchmark.context_length, 1).to(device=device)
            next_snr = schedule[step_idx + 1].expand(current_batch, benchmark.context_length, 1).to(device=device)
            emb_hat, _logits_ext, logits = model(noisy, snr)
            tokens = sample_from_logits(logits, temperature=temperature, top_k=top_k, generator=generator)
            mean = posterior_mean_from_logits(logits, base_model.x_values.weight.to(device=device, dtype=noisy.dtype))
            denoised = emb_hat if emb_hat is not None else mean
            noise = torch.randn(denoised.shape, generator=generator, dtype=denoised.dtype).to(device=device)
            noisy = denoised + noise / torch.sqrt(torch.clamp(next_snr, min=1.0e-12))
        if tokens is None:
            raise RuntimeError("IMMSE sampler did not run any denoising steps.")
        generated.append(benchmark.decode_token_sequences(tokens.cpu()))
    return torch.cat(generated, dim=0)[: int(count)].clamp(0.0, 1.0)


def sample_images_for_cfg(model: torch.nn.Module, benchmark: Any, cfg: Any, **kwargs: Any) -> torch.Tensor:
    kind = str(cfg.kind)
    if kind == "img_exit":
        kwargs.pop("hold_fraction", None)
        return sample_exit_images(model, benchmark, **kwargs)
    if kind == "img_qsc":
        return sample_qsc_images(model, benchmark, cfg=cfg, **kwargs)
    if kind == "img_immse":
        kwargs.pop("hold_fraction", None)
        return sample_immse_images(model, benchmark, cfg=cfg, **kwargs)
    raise ValueError(f"Unsupported kind={kind!r}.")


def _ksc_lambda_from_t(t: torch.Tensor, alphabet_size: int) -> torch.Tensor:
    """Map KSC pass-through t to the non-trivial eigenvalue λ in [-1/(K-1), 1]."""
    K = float(alphabet_size)
    return (K * t - 1.0) / (K - 1.0)


def _ksc_t_from_lambda(lam: torch.Tensor, alphabet_size: int) -> torch.Tensor:
    """Inverse of _ksc_lambda_from_t."""
    K = float(alphabet_size)
    return (1.0 + (K - 1.0) * lam) / K


def _make_ksc_tbar_schedule(
    *,
    alphabet_size: int,
    steps: int,
    t_min: float,
    t_max: float,
    schedule: str,
) -> torch.Tensor:
    """Create a cumulative KSC schedule t̄[0..T] for x0->x_t.

    - `linear_tbar`: linearly interpolate t̄ between t_max -> t_min.
    - `geometric_lambda`: linearly interpolate log λ̄, which yields roughly-constant per-step noise in λ-space.
    - `linear_beta`: linearly interpolate per-step β (with Qt = (1-β)I + β*U_K) in λ-space,
      then accumulate λ̄. This avoids extreme final-step noise when targeting t_min=1/K.
    """
    if int(steps) <= 0:
        raise ValueError("steps must be > 0.")
    K = int(alphabet_size)
    if schedule == "linear_tbar":
        return torch.linspace(float(t_max), float(t_min), steps=int(steps) + 1, dtype=torch.float32)
    if schedule != "geometric_lambda":
        if schedule != "linear_beta":
            raise ValueError(f"Unsupported schedule={schedule!r}.")
        # Per-step beta schedule: Qt = (1-β)I + β*U_K (uniform mixing).
        # In λ-space this is λ_step = 1-β. Cumulative λ̄_t = ∏_s (1-β_s).
        # We choose β to go from small->large over time, then compute λ̄ and map to t̄.
        beta_start = 1.0e-4
        beta_end = 2.0e-2
        beta = torch.linspace(beta_start, beta_end, steps=int(steps), dtype=torch.float32)
        lam_step = 1.0 - beta
        lam_bar = torch.cumprod(torch.cat([torch.ones(1, dtype=torch.float32), lam_step], dim=0), dim=0)
        return _ksc_t_from_lambda(lam_bar, K).clamp(min=1.0 / float(K), max=1.0)
    lam0 = float(_ksc_lambda_from_t(torch.tensor(float(t_max)), K).item())
    lamT = float(_ksc_lambda_from_t(torch.tensor(float(t_min)), K).item())
    lam0 = max(lam0, 1.0e-12)
    lamT = max(lamT, 1.0e-12)
    log_lam = torch.linspace(math.log(lam0), math.log(lamT), steps=int(steps) + 1, dtype=torch.float32)
    lam_bar = torch.exp(log_lam)
    return _ksc_t_from_lambda(lam_bar, K).clamp(min=1.0 / float(K), max=1.0)


def qsc_d3pm_posterior_logits_from_x0_posterior(
    *,
    p_x0: torch.Tensor,
    x_t: torch.Tensor,
    t_cur: float,
    t_prev: float,
    t_step: float,
    alphabet_size: int,
    eps: float = 1e-20,
) -> torch.Tensor:
    """Compute D3PM q(x_{t-1}|x_t) logits from p_theta(x0|x_t).

    The model posterior `p_x0` is mixed with the exact KSC posterior
    q(x_{t-1}|x_t,x0) ∝ Q_t[x_{t-1}, x_t] Q̄_{t-1}[x0, x_{t-1}] / Q̄_t[x0, x_t].
    """
    K = int(alphabet_size)
    if K <= 1:
        raise ValueError(f"alphabet_size must be > 1, got {alphabet_size}.")
    idx = x_t.unsqueeze(-1)

    c_cur = (1.0 - float(t_cur)) / float(K - 1)
    denom = torch.full_like(p_x0, float(c_cur))
    denom.scatter_(-1, idx, float(t_cur))
    weights = p_x0 / denom.clamp_min(float(eps))

    c_prev = (1.0 - float(t_prev)) / float(K - 1)
    weighted_mass = weights.sum(dim=-1, keepdim=True)
    propagated = float(c_prev) * weighted_mass + (float(t_prev) - float(c_prev)) * weights

    c_step = (1.0 - float(t_step)) / float(K - 1)
    p_prev_unnorm = propagated * float(c_step)
    propagated_at_xt = propagated.gather(-1, idx)
    p_prev_unnorm.scatter_add_(-1, idx, (float(t_step) - float(c_step)) * propagated_at_xt)
    return torch.log(p_prev_unnorm.clamp_min(float(eps)))


@torch.no_grad()
def sample_qsc_images_d3pm_posterior(
    model: torch.nn.Module,
    benchmark: Any,
    *,
    cfg: Any,
    count: int,
    batch_size: int,
    num_diff_steps: int,
    temperature: float,
    top_k: int,
    device: torch.device,
    generator: torch.Generator,
    eps: float = 1e-20,
    schedule: str = "geometric_lambda",
) -> torch.Tensor:
    """D3PM-style reverse posterior sampling for the K-ary symmetric channel (QSC).

    Implements the per-token reverse step from the user's pseudocode:

      m = Q̄_{t-1} @ p(x0)         # propagate x0 -> x_{t-1}
      p(x_{t-1}|x_t) ∝ Q_t[:,x_t] ⊙ m

    Here Q̄_{t} is the cumulative KSC kernel from x0 to x_t with marginal pass-through
    rate t̄, and Q_t is the per-step kernel chosen so that the Markov chain marginals
    match the provided t̄ schedule.
    """
    base_model = unwrap_model(model)
    K = int(base_model.vocab_size)
    generated: list[torch.Tensor] = []

    # We interpret the schedule as *marginal* KSC rates t̄_t from x0 -> x_t.
    # Forward (corruption) runs from clean->noise: high t̄ -> low t̄.
    t_min = float(getattr(cfg.eval, "t_min", getattr(cfg.qsc, "train_t_min", 1.0 / float(K))))
    t_max = float(getattr(cfg.eval, "t_max", getattr(cfg.qsc, "train_t_max", 0.9999)))
    tbar = _make_ksc_tbar_schedule(
        alphabet_size=K,
        steps=int(num_diff_steps),
        t_min=t_min,
        t_max=t_max,
        schedule=str(schedule),
    )  # [T+1]
    lam_bar = _ksc_lambda_from_t(tbar, K).clamp(min=0.0, max=1.0)  # monotone from ~1 -> 0

    # Per-step KSC kernels Q_t such that lam_bar[t] = lam_step[t] * lam_bar[t-1].
    lam_step = torch.zeros(int(num_diff_steps) + 1, dtype=torch.float32)
    for t_idx in range(1, int(num_diff_steps) + 1):
        denom = float(lam_bar[t_idx - 1].item())
        lam_step[t_idx] = 0.0 if denom <= 0.0 else float(lam_bar[t_idx].item()) / denom
    t_step = _ksc_t_from_lambda(lam_step, K).clamp(min=1.0 / float(K), max=1.0)  # [T+1], t_step[0] unused

    while sum(int(chunk.shape[0]) for chunk in generated) < int(count):
        current_batch = min(int(batch_size), int(count) - sum(int(chunk.shape[0]) for chunk in generated))

        # x_T ~ Uniform over symbols (matches t̄_T ~= 1/K).
        x_t = torch.randint(K, (current_batch, benchmark.context_length), generator=generator, dtype=torch.long).to(device=device)

        # Reverse t = T..1, sample x_{t-1} given x_t.
        for t_idx in range(int(num_diff_steps), 0, -1):
            t_cur = float(tbar[t_idx].item())
            t_prev = float(tbar[t_idx - 1].item())
            t_step_val = float(t_step[t_idx].item())

            t_cur_batch = torch.full((current_batch,), t_cur, dtype=torch.float32, device=device)
            tilt = log_ksc_kernel(x_t, t_cur_batch, K)
            outputs = model.forward_outputs(x_t, timesteps=t_cur_batch, channel_tilt=tilt) if hasattr(model, "forward_outputs") else None
            logits = outputs.logits if outputs is not None else model(x_t, timesteps=t_cur_batch) + tilt
            p_x0 = torch.softmax(logits, dim=-1)  # [B, S, K]

            logits_prev = qsc_d3pm_posterior_logits_from_x0_posterior(
                p_x0=p_x0,
                x_t=x_t,
                t_cur=t_cur,
                t_prev=t_prev,
                t_step=t_step_val,
                alphabet_size=K,
                eps=float(eps),
            )
            x_t = sample_from_logits(logits_prev, temperature=temperature, top_k=top_k, generator=generator)

        generated.append(benchmark.decode_token_sequences(x_t.cpu()))

    return torch.cat(generated, dim=0)[: int(count)].clamp(0.0, 1.0)


@torch.no_grad()
def sample_qsc_images_d3pm_reverse_param(
    model: torch.nn.Module,
    benchmark: Any,
    *,
    cfg: Any,
    count: int,
    batch_size: int,
    num_diff_steps: int,
    temperature: float,
    top_k: int,
    device: torch.device,
    generator: torch.Generator,
    eps: float = 1e-20,
    schedule: str = "geometric_lambda",
    xT_from_model_prior: bool = True,
) -> torch.Tensor:
    """D3PM-style sampler using the reverse parameterization p_theta(x_{t-1}|x_t).

    This matches the common D3PM implementation pattern:
      1) Predict x0 distribution:  x0_guess = p_theta(x0 | x_t, t)
      2) Compute q-posterior logits using (x0_guess, x_t, Q_t, Q̄_{t-1})
      3) Sample x_{t-1} from that categorical distribution.

    NOTE: For QSC-trained models, starting from x_T ~ Uniform often looks bad because
    the model may not represent the unconditional prior strongly. This function can
    instead sample x_T by first sampling x0 from the model prior at t≈t_max and then
    corrupting to t̄_T, controlled by `xT_from_model_prior`.
    """
    base_model = unwrap_model(model)
    K = int(base_model.vocab_size)
    generated: list[torch.Tensor] = []

    t_min = float(getattr(cfg.eval, "t_min", getattr(cfg.qsc, "train_t_min", 1.0 / float(K))))
    t_max = float(getattr(cfg.eval, "t_max", getattr(cfg.qsc, "train_t_max", 0.9999)))
    tbar = _make_ksc_tbar_schedule(
        alphabet_size=K,
        steps=int(num_diff_steps),
        t_min=t_min,
        t_max=t_max,
        schedule=str(schedule),
    )  # [T+1], clean->noise
    lam_bar = _ksc_lambda_from_t(tbar, K).clamp(min=0.0, max=1.0)

    lam_step = torch.zeros(int(num_diff_steps) + 1, dtype=torch.float32)
    for t_idx in range(1, int(num_diff_steps) + 1):
        denom = float(lam_bar[t_idx - 1].item())
        lam_step[t_idx] = 0.0 if denom <= 0.0 else float(lam_bar[t_idx].item()) / denom
    t_step = _ksc_t_from_lambda(lam_step, K).clamp(min=1.0 / float(K), max=1.0)

    while sum(int(chunk.shape[0]) for chunk in generated) < int(count):
        current_batch = min(int(batch_size), int(count) - sum(int(chunk.shape[0]) for chunk in generated))

        if bool(xT_from_model_prior):
            # Approximate model prior sampling via a single denoise at high t̄ (near-clean),
            # then corrupt down to t̄_T. This bootstraps some structure into x_T.
            t0 = float(tbar[0].item())
            x0_seed = torch.randint(K, (current_batch, benchmark.context_length), generator=generator, dtype=torch.long).to(device=device)
            t0_batch = torch.full((current_batch,), t0, dtype=torch.float32, device=device)
            tilt0 = log_ksc_kernel(x0_seed, t0_batch, K)
            outputs0 = model.forward_outputs(x0_seed, timesteps=t0_batch, channel_tilt=tilt0) if hasattr(model, "forward_outputs") else None
            logits0 = outputs0.logits if outputs0 is not None else model(x0_seed, timesteps=t0_batch) + tilt0
            x0 = sample_from_logits(logits0, temperature=temperature, top_k=top_k, generator=generator)
            x_t, _ = apply_ksc_noise(x0, torch.full((current_batch,), float(tbar[-1].item()), device=device), K, generator=generator)
        else:
            # Classical D3PM: x_T ~ Uniform.
            x_t = torch.randint(K, (current_batch, benchmark.context_length), generator=generator, dtype=torch.long).to(device=device)

        for t_idx in range(int(num_diff_steps), 0, -1):
            t_cur = float(tbar[t_idx].item())
            t_prev = float(tbar[t_idx - 1].item())
            t_step_val = float(t_step[t_idx].item())

            t_cur_batch = torch.full((current_batch,), t_cur, dtype=torch.float32, device=device)
            tilt = log_ksc_kernel(x_t, t_cur_batch, K)
            outputs = model.forward_outputs(x_t, timesteps=t_cur_batch, channel_tilt=tilt) if hasattr(model, "forward_outputs") else None
            logits = outputs.logits if outputs is not None else model(x_t, timesteps=t_cur_batch) + tilt
            x0_guess = torch.softmax(logits, dim=-1)  # [B,S,K]

            logits_prev = qsc_d3pm_posterior_logits_from_x0_posterior(
                p_x0=x0_guess,
                x_t=x_t,
                t_cur=t_cur,
                t_prev=t_prev,
                t_step=t_step_val,
                alphabet_size=K,
                eps=float(eps),
            )
            x_t = sample_from_logits(logits_prev, temperature=temperature, top_k=top_k, generator=generator)

        generated.append(benchmark.decode_token_sequences(x_t.cpu()))

    return torch.cat(generated, dim=0)[: int(count)].clamp(0.0, 1.0)


@torch.no_grad()
def sample_qsc_images_reverse_kernel(
    model: torch.nn.Module,
    benchmark: Any,
    *,
    cfg: Any,
    count: int,
    batch_size: int,
    num_diff_steps: int,
    temperature: float,
    top_k: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    """QSC sampler that explicitly simulates a learned reverse kernel on observations.

    At each step we compute p_theta(x|y,t) via logit-tilting, then define a reverse
    kernel on the next observation y' at t' as:
        q(y') = t' * p(x=y') + (1-t')/(K-1) * (1 - p(x=y'))
    and sample y' ~ q. After the final step, decode a clean sample from p_theta(x|y,t_max).
    """
    base_model = unwrap_model(model)
    K = int(base_model.vocab_size)
    generated: list[torch.Tensor] = []
    t_min = float(getattr(cfg.eval, "t_min", getattr(cfg.qsc, "train_t_min", 1.0 / float(K))))
    t_max = float(getattr(cfg.eval, "t_max", getattr(cfg.qsc, "train_t_max", 0.9999)))
    schedule = torch.linspace(t_min, t_max, steps=int(num_diff_steps) + 1, dtype=torch.float32)
    noise_coeff = (1.0 - schedule) / float(max(1, K - 1))  # shape [N+1]

    while sum(int(chunk.shape[0]) for chunk in generated) < int(count):
        current_batch = min(int(batch_size), int(count) - sum(int(chunk.shape[0]) for chunk in generated))
        # Start from maximally noisy observations (uniform tokens).
        y = torch.randint(K, (current_batch, benchmark.context_length), generator=generator, dtype=torch.long).to(device=device)

        for step_idx in range(int(num_diff_steps)):
            t = schedule[step_idx].expand(current_batch).to(device=device)
            t_next = schedule[step_idx + 1].expand(current_batch).to(device=device)

            tilt_t = log_ksc_kernel(y, t, K)
            outputs = model.forward_outputs(y, timesteps=t, channel_tilt=tilt_t) if hasattr(model, "forward_outputs") else None
            logits = outputs.logits if outputs is not None else model(y, timesteps=t) + tilt_t
            p = torch.softmax(logits, dim=-1)

            # Reverse kernel on observations: q(y') = t_next*p + ((1-t_next)/(K-1))*(1-p)
            q = t_next.view(current_batch, 1, 1) * p + noise_coeff[step_idx + 1].view(1, 1, 1).to(device=device, dtype=p.dtype) * (1.0 - p)
            y = sample_from_logits(q, temperature=temperature, top_k=top_k, generator=generator)

        # Final clean sample from p_theta(x|y,t_max)
        t_final = schedule[-1].expand(current_batch).to(device=device)
        tilt_final = log_ksc_kernel(y, t_final, K)
        outputs = model.forward_outputs(y, timesteps=t_final, channel_tilt=tilt_final) if hasattr(model, "forward_outputs") else None
        logits_final = outputs.logits if outputs is not None else model(y, timesteps=t_final) + tilt_final
        x = sample_from_logits(logits_final, temperature=temperature, top_k=top_k, generator=generator)
        generated.append(benchmark.decode_token_sequences(x.cpu()))

    return torch.cat(generated, dim=0)[: int(count)].clamp(0.0, 1.0)


@torch.no_grad()
def sample_qsc_images_late_freeze(
    model: torch.nn.Module,
    benchmark: Any,
    *,
    cfg: Any,
    count: int,
    batch_size: int,
    num_diff_steps: int,
    temperature: float,
    top_k: int,
    freeze_start_t: float = 0.7,
    freeze_power: float = 2.0,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    """QSC sampler with a masked-like monotone freeze that only activates at high t.

    Rationale: freezing too early locks in garbage. Here we only begin freezing once
    t exceeds freeze_start_t, and ramp the freeze probability up towards 1 as t->t_max.
    """
    base_model = unwrap_model(model)
    K = int(base_model.vocab_size)
    generated: list[torch.Tensor] = []
    t_min = float(getattr(cfg.eval, "t_min", getattr(cfg.qsc, "train_t_min", 1.0 / float(K))))
    t_max = float(getattr(cfg.eval, "t_max", getattr(cfg.qsc, "train_t_max", 0.9999)))
    schedule = torch.linspace(t_min, t_max, steps=int(num_diff_steps) + 1, dtype=torch.float32)
    freeze_start_t = float(freeze_start_t)

    while sum(int(chunk.shape[0]) for chunk in generated) < int(count):
        current_batch = min(int(batch_size), int(count) - sum(int(chunk.shape[0]) for chunk in generated))
        clean = torch.randint(K, (current_batch, benchmark.context_length), generator=generator, dtype=torch.long).to(device=device)
        observed = clean
        frozen = torch.zeros_like(clean, dtype=torch.bool)

        for step_idx in range(int(num_diff_steps)):
            t_val = float(schedule[step_idx].item())
            t = schedule[step_idx].expand(current_batch).to(device=device)
            next_t = schedule[step_idx + 1].expand(current_batch).to(device=device)

            kernel = log_ksc_kernel(observed, t, K)
            outputs = model.forward_outputs(observed, timesteps=t, channel_tilt=kernel) if hasattr(model, "forward_outputs") else None
            logits = outputs.logits if outputs is not None else model(observed, timesteps=t) + kernel

            proposed = sample_from_logits(logits, temperature=float(temperature), top_k=top_k, generator=generator)
            clean = torch.where(frozen, clean, proposed)

            # Freeze only at high t, ramping up toward the end.
            if t_val >= freeze_start_t:
                progress = (t_val - freeze_start_t) / max(1e-6, (t_max - freeze_start_t))
                freeze_prob = max(0.0, min(1.0, progress)) ** float(freeze_power)
                if freeze_prob > 0.0:
                    uniforms = torch.rand(clean.shape, generator=generator, dtype=torch.float32).to(device=device)
                    new_freeze = (uniforms < freeze_prob) & (~frozen)
                    frozen = frozen | new_freeze

            noisy, _pass_mask = apply_ksc_noise(clean, next_t, K, generator=generator)
            observed = torch.where(frozen, clean, noisy)

        generated.append(benchmark.decode_token_sequences(clean.cpu()))
    return torch.cat(generated, dim=0)[: int(count)].clamp(0.0, 1.0)


@torch.no_grad()
def sample_qsc_images_no_renoise(
    model: torch.nn.Module,
    benchmark: Any,
    *,
    cfg: Any,
    count: int,
    batch_size: int,
    num_diff_steps: int,
    temperature: float,
    top_k: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    """QSC sampler variant that does NOT re-apply channel noise between denoising steps.

    Each step:
      - compute tilted posterior logits from current tokens treated as 'observed'
      - sample clean tokens
      - feed them back as the next 'observed' tokens (no apply_ksc_noise)
    """
    base_model = unwrap_model(model)
    K = int(base_model.vocab_size)
    generated: list[torch.Tensor] = []
    t_min = float(getattr(cfg.eval, "t_min", getattr(cfg.qsc, "train_t_min", 1.0 / float(K))))
    t_max = float(getattr(cfg.eval, "t_max", getattr(cfg.qsc, "train_t_max", 0.9999)))
    schedule = torch.linspace(t_min, t_max, steps=int(num_diff_steps) + 1, dtype=torch.float32)

    while sum(int(chunk.shape[0]) for chunk in generated) < int(count):
        current_batch = min(int(batch_size), int(count) - sum(int(chunk.shape[0]) for chunk in generated))
        observed = torch.randint(K, (current_batch, benchmark.context_length), generator=generator, dtype=torch.long).to(device=device)
        clean = observed
        for step_idx in range(int(num_diff_steps)):
            t = schedule[step_idx].expand(current_batch).to(device=device)
            kernel = log_ksc_kernel(observed, t, K)
            outputs = model.forward_outputs(observed, timesteps=t, channel_tilt=kernel) if hasattr(model, "forward_outputs") else None
            logits = outputs.logits if outputs is not None else model(observed, timesteps=t) + kernel
            clean = sample_from_logits(logits, temperature=temperature, top_k=top_k, generator=generator)
            observed = clean
        generated.append(benchmark.decode_token_sequences(clean.cpu()))
    return torch.cat(generated, dim=0)[: int(count)].clamp(0.0, 1.0)


def images_to_uint8_nchw(images: torch.Tensor) -> torch.Tensor:
    images_t = torch.as_tensor(images, dtype=torch.float32, device="cpu").clamp(0.0, 1.0)
    if images_t.ndim != 4 or int(images_t.shape[-1]) != 3:
        raise ValueError(f"Expected images [N, H, W, 3], got {tuple(images_t.shape)}")
    return torch.clamp(torch.round(images_t * 255.0), min=0.0, max=255.0).to(dtype=torch.uint8).permute(0, 3, 1, 2).contiguous()


def _repeat_to_count(images: torch.Tensor, count: int) -> tuple[torch.Tensor, bool]:
    if int(images.shape[0]) >= int(count):
        return images[: int(count)], False
    repeats = (int(count) + int(images.shape[0]) - 1) // int(images.shape[0])
    return images.repeat((repeats, 1, 1, 1))[: int(count)], True


def compute_generation_metrics(generated_images: torch.Tensor, real_images: torch.Tensor, *, sample_count: int, batch_size: int, device: torch.device) -> dict[str, float | bool]:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore

    real_eval, repeated_real = _repeat_to_count(real_images, int(sample_count))
    real_uint8 = images_to_uint8_nchw(real_eval)
    gen_uint8 = images_to_uint8_nchw(generated_images)
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    inception = InceptionScore(normalize=False, splits=max(1, min(10, int(gen_uint8.shape[0])))).to(device)
    for start in range(0, int(real_uint8.shape[0]), int(batch_size)):
        fid.update(real_uint8[start : start + int(batch_size)].to(device), real=True)
    for start in range(0, int(gen_uint8.shape[0]), int(batch_size)):
        batch = gen_uint8[start : start + int(batch_size)].to(device)
        fid.update(batch, real=False)
        inception.update(batch)
    is_mean, is_std = inception.compute()
    return {
        "fid": float(fid.compute().item()),
        "inception_score_mean": float(is_mean.item()),
        "inception_score_std": float(is_std.item()),
        "real_images_repeated": bool(repeated_real),
    }


def _load_wandb_run_for_logging(run: str, *, cfg: Any, entity: str | None, project: str | None) -> Any:
    import wandb

    run_path = _resolve_run_path(run, cfg=cfg, entity=entity, project=project)
    resolved_entity, resolved_project, resolved_id = run_path.split("/")
    return wandb.init(
        entity=resolved_entity,
        project=resolved_project,
        id=resolved_id,
        resume="allow",
        mode=str(getattr(cfg.logging, "mode", "online")),
        settings=wandb.Settings(start_method=None),
    )


def _wandb_log_metrics(wb_run: Any, row: dict[str, Any]) -> None:
    step = int(row["step"])
    payload = {
        f"eval_fid/{step}": float(row["fid"]),
        f"eval_inception_score_mean/{step}": float(row["inception_score_mean"]),
        f"eval_inception_score_std/{step}": float(row["inception_score_std"]),
        "eval_fid": float(row["fid"]),
        "eval_inception_score_mean": float(row["inception_score_mean"]),
        "eval_inception_score_std": float(row["inception_score_std"]),
    }
    wb_run.log(payload, step=step)


def main(argv: list[str] | None = None) -> int:
    configure_environment()
    args = build_arg_parser().parse_args(argv)
    set_global_seed(int(args.seed), deterministic=False)
    device = get_best_device(str(args.device))
    cache_dir = ensure_dir(Path(args.cache_dir).expanduser() if args.cache_dir is not None else _default_cache_dir(args.run))
    output_dir = ensure_dir(Path(args.output_dir).expanduser() if args.output_dir is not None else _default_output_dir(args.run))
    config_path, checkpoints, _api_run = ensure_cached_run_files(run=args.run, cache_dir=cache_dir, entity=args.entity, project=args.project, max_checkpoints=args.max_checkpoints)
    cfg = OmegaConf.load(config_path)
    if str(cfg.kind) not in {"img_exit", "img_qsc", "img_immse"}:
        raise ValueError(f"Unsupported image generation eval kind={cfg.kind!r}; expected img_exit, img_qsc, or img_immse.")
    benchmark = load_img_benchmark(cfg.benchmark, project_root=find_repo_root(APP_ROOT))
    real_images = benchmark.split_images(str(args.split))
    if real_images is None and not bool(args.skip_metrics):
        raise ValueError(f"Benchmark split {args.split!r} does not include real images for FID.")
    if int(args.sample_count) <= 0 or int(args.batch_size) <= 0 or int(args.num_diff_steps) <= 0:
        raise ValueError("sample-count, batch-size, and num-diff-steps must be positive.")
    wb_run = _load_wandb_run_for_logging(args.run, cfg=cfg, entity=args.entity, project=args.project) if bool(args.log_wandb) else None

    rows: list[dict[str, Any]] = []
    sample_dir = ensure_dir(output_dir / "samples")
    tensor_dir = ensure_dir(output_dir / "generated_tensors") if bool(args.save_generated) else None
    start_time = time.perf_counter()
    for checkpoint in checkpoints:
        step = checkpoint_step(checkpoint)
        print(f"[eval] checkpoint={checkpoint} step={step}", flush=True)
        model = build_image_model(cfg, vocab_size=benchmark.vocab_size, context_length=benchmark.context_length)
        model.load_state_dict(load_model_state_dict(checkpoint))
        model = model.to(device).eval()
        generator = torch.Generator(device=device)
        generator.manual_seed(int(args.seed) + int(step))
        generated = sample_images_for_cfg(
            model,
            benchmark,
            cfg,
            count=int(args.sample_count),
            batch_size=int(args.batch_size),
            num_diff_steps=int(args.num_diff_steps),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            hold_fraction=float(args.hold_fraction),
            device=device,
            generator=generator,
        )
        grid_path = sample_dir / f"{checkpoint.stem}_samples.png"
        save_image_grid(grid_path, rows=[("samples", generated[: min(16, int(generated.shape[0]))])], max_images=min(16, int(generated.shape[0])))
        if tensor_dir is not None:
            torch.save(generated, tensor_dir / f"{checkpoint.stem}_generated.pt")
        metric_payload: dict[str, float | bool]
        if bool(args.skip_metrics):
            metric_payload = {"fid": float("nan"), "inception_score_mean": float("nan"), "inception_score_std": float("nan"), "real_images_repeated": False}
        else:
            assert real_images is not None
            metric_payload = compute_generation_metrics(generated, real_images, sample_count=int(args.sample_count), batch_size=int(args.batch_size), device=device)
        row = {
            "run": str(args.run),
            "kind": str(cfg.kind),
            "checkpoint": str(checkpoint),
            "step": int(step),
            "sample_count": int(args.sample_count),
            "num_diff_steps": int(args.num_diff_steps),
            "batch_size": int(args.batch_size),
            "split": str(args.split),
            "temperature": float(args.temperature),
            "top_k": int(args.top_k),
            **metric_payload,
            "sample_grid": str(grid_path),
        }
        rows.append(row)
        save_csv(output_dir / "checkpoint_metrics.csv", rows)
        save_json(
            output_dir / "checkpoint_metrics.json",
            {
                "run": str(args.run),
                "config": str(config_path),
                "cache_dir": str(cache_dir),
                "output_dir": str(output_dir),
                "kind": str(cfg.kind),
                "benchmark": benchmark.benchmark_metadata(),
                "device": str(device),
                "elapsed_seconds": float(time.perf_counter() - start_time),
                "checkpoints": rows,
            },
        )
        if bool(args.log_wandb) and wb_run is not None and not bool(args.skip_metrics):
            _wandb_log_metrics(wb_run, row)
    if wb_run is not None and hasattr(wb_run, "finish"):
        wb_run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
