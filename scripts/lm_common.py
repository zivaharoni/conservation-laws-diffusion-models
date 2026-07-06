from __future__ import annotations

import csv
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import OmegaConf

from experiments.common.config import resolve_config_name_and_overrides
from experiments.common.io import deterministic_output_dir
from experiments.common.wandb_logger import WandbLogger
from src.lm_distributed import unwrap_model


def find_app_root(path: Path) -> Path:
    current = path.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "configs").is_dir() and (candidate / "scripts").is_dir() and (candidate / "src").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find app root from {path}.")


def find_repo_root(app_root: Path) -> Path:
    app_root = app_root.resolve()
    if app_root.name == "code":
        return app_root.parent
    return app_root


def compose_lm_config(*, project_root: Path, argv: list[str]):
    config_name, overrides = resolve_config_name_and_overrides(argv)
    config_dir = project_root / "configs"
    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = hydra.compose(config_name=config_name, overrides=overrides)
    OmegaConf.set_struct(cfg, True)
    return config_name, overrides, cfg


def resolve_project_path(project_root: Path, value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return path


def resolve_output_dir(*, project_root: Path, cfg, config_name: str, overrides: list[str]) -> tuple[str, Path]:
    from experiments.common.repro import make_run_id

    results_root = str((project_root / str(cfg.paths.results_root)).resolve())
    cfg.paths.results_root = results_root
    run_id = str(getattr(cfg.logging, "run_id", None) or make_run_id(config_name, overrides))
    output_dir = deterministic_output_dir(project_root, cfg.paths.results_root, cfg.name, run_id)
    return run_id, output_dir


def count_parameters(model: torch.nn.Module) -> int:
    return sum(int(param.numel()) for param in unwrap_model(model).parameters())


def _load_compatible_model_state(model: torch.nn.Module, state_dict: Mapping[str, torch.Tensor]) -> dict[str, int]:
    base_model = unwrap_model(model)
    current_state = base_model.state_dict()
    filtered_state: dict[str, torch.Tensor] = {}
    skipped_missing = 0
    skipped_shape = 0
    for key, value in state_dict.items():
        if key not in current_state:
            skipped_missing += 1
            continue
        if tuple(current_state[key].shape) != tuple(value.shape):
            skipped_shape += 1
            continue
        filtered_state[key] = value
    base_model.load_state_dict(filtered_state, strict=False)
    return {"loaded": len(filtered_state), "skipped_missing": skipped_missing, "skipped_shape": skipped_shape}


def load_checkpoint_if_requested(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str | Path | None,
    project_root: Path,
) -> tuple[int, int]:
    resolved = resolve_project_path(project_root, checkpoint_path)
    if resolved is None:
        return 1, 0
    if not resolved.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {resolved}")
    payload = torch.load(resolved, map_location="cpu")
    model_state = payload["model_state_dict"]
    try:
        unwrap_model(model).load_state_dict(model_state)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    except (RuntimeError, ValueError) as exc:
        stats = _load_compatible_model_state(model, model_state)
        print(
            f"[setup] checkpoint {resolved} is not strictly compatible; loaded compatible model tensors only "
            f"(loaded={stats['loaded']} skipped_missing={stats['skipped_missing']} skipped_shape={stats['skipped_shape']}). "
            f"Optimizer state was not restored. Original error: {exc}",
            flush=True,
        )
    return int(payload.get("step", 0)) + 1, int(payload.get("tokens_seen", 0))


def load_initial_weights_if_requested(
    *,
    model: torch.nn.Module,
    checkpoint_path: str | Path | None,
    project_root: Path,
    strict: bool = True,
) -> Path | None:
    resolved = resolve_project_path(project_root, checkpoint_path)
    if resolved is None:
        return None
    if not resolved.is_file():
        raise FileNotFoundError(f"Initial weights checkpoint not found: {resolved}")
    payload = torch.load(resolved, map_location="cpu")
    state_dict: Mapping[str, torch.Tensor] | None = None
    if isinstance(payload, Mapping):
        for key in ("model_state_dict", "state_dict"):
            candidate = payload.get(key)
            if isinstance(candidate, Mapping):
                state_dict = candidate
                break
        if state_dict is None and all(isinstance(name, str) for name in payload):
            state_dict = payload
    if state_dict is None:
        raise TypeError(f"Could not extract a model state dict from {resolved}")
    unwrap_model(model).load_state_dict(state_dict, strict=bool(strict))
    return resolved


def save_training_checkpoint(
    *,
    output_dir: Path,
    checkpoint_dir_name: str,
    filename_template: str,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    tokens_seen: int,
    extra_state: Mapping[str, Any] | None = None,
) -> Path:
    checkpoint_dir = output_dir / checkpoint_dir_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / str(filename_template).format(step=int(step))
    payload: dict[str, Any] = {
        "step": int(step),
        "tokens_seen": int(tokens_seen),
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if extra_state:
        payload.update(dict(extra_state))
    torch.save(payload, path)
    return path


def runtime_metrics(*, start_time: float, device: torch.device) -> dict[str, Any]:
    elapsed = time.perf_counter() - start_time
    metrics: dict[str, Any] = {
        "elapsed_seconds": float(elapsed),
        "device": str(device),
    }
    if device.type == "cuda":
        metrics["peak_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
        metrics["peak_memory_gb"] = float(torch.cuda.max_memory_allocated(device) / (1024**3))
    return metrics


def load_history_prefix(path: str | Path, *, start_step: int) -> list[dict[str, str]]:
    resolved = Path(path)
    if not resolved.is_file():
        return []
    with resolved.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if int(start_step) <= 1:
        return []
    return [row for row in rows if row.get("step") not in (None, "") and int(row["step"]) < int(start_step)]


def maybe_reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def lm_embedding_norm(model: torch.nn.Module) -> float:
    base_model = unwrap_model(model)
    if hasattr(base_model, "x_values"):
        weight = base_model.x_values.weight
    elif hasattr(base_model, "token_embedding"):
        weight = base_model.token_embedding.weight
    else:
        return float("nan")
    return float(weight.detach().norm(dim=-1).mean().item())


def maybe_apply_lm_lr_schedule(cfg, *, step: int, optimizer: torch.optim.Optimizer) -> float:
    schedule_cfg = getattr(cfg.train, "lr_schedule", None)
    if schedule_cfg is None or not bool(schedule_cfg.enabled):
        return float(optimizer.param_groups[0]["lr"])

    warmup_steps = max(0, int(schedule_cfg.warmup_steps))
    min_lr = float(schedule_cfg.min_lr)
    base_lr = float(cfg.train.lr)
    schedule_name = str(getattr(schedule_cfg, "name", "cosine")).lower()

    if schedule_name in {"constant", "flat"}:
        if warmup_steps > 0 and step <= warmup_steps:
            new_lr = base_lr * (float(step) / float(warmup_steps))
        else:
            new_lr = base_lr
    elif schedule_name in {"inverse_sqrt", "inverse-square-root", "rsqrt"}:
        if warmup_steps <= 0:
            raise ValueError("inverse_sqrt lr schedule requires train.lr_schedule.warmup_steps > 0.")
        if step <= warmup_steps:
            new_lr = base_lr * (float(step) / float(warmup_steps))
        else:
            new_lr = base_lr * math.sqrt(float(warmup_steps) / float(step))
    else:
        total_steps = max(1, int(cfg.train.n_steps))
        if warmup_steps > 0 and step <= warmup_steps:
            new_lr = base_lr * (float(step) / float(warmup_steps))
        else:
            cosine_steps = max(1, total_steps - warmup_steps)
            cosine_progress = min(1.0, max(0.0, float(step - warmup_steps) / float(cosine_steps)))
            cosine = 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
            new_lr = min_lr + (base_lr - min_lr) * cosine
    new_lr = max(min_lr, new_lr)

    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
    return float(new_lr)


def prefixed_scalar_metrics(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, (int, float, bool, str)):
            payload[f"{prefix}/{key}"] = value
    return payload


def split_exists(benchmark: Any, split: str) -> bool:
    try:
        benchmark.split_tokens(str(split))
    except Exception:
        return False
    return True


def resolve_valid_split_name(benchmark: Any) -> str:
    for candidate in ("val", "valid"):
        if split_exists(benchmark, candidate):
            return candidate
    raise ValueError("Benchmark does not expose a validation split.")


def resolve_test_split_name(benchmark: Any) -> str:
    if split_exists(benchmark, "test"):
        return "test"
    return resolve_valid_split_name(benchmark)


def eval_generate_curves(cfg: Any) -> bool:
    return bool(getattr(getattr(cfg, "eval", None), "generate_curves", True))


def eval_log_conservation_integral(cfg: Any) -> bool:
    return bool(getattr(getattr(cfg, "eval", None), "log_conservation_integral", False))


def conservation_law_nats(*, objective: str, metrics: Mapping[str, Any]) -> float:
    objective_name = str(objective).lower()
    if objective_name == "causal_ce":
        value = metrics.get("nats_per_token")
    else:
        value = metrics.get(
            "integral_conservation_law_nats",
            metrics.get("curve_integral_nats", metrics.get("nats_per_token")),
        )
    if value is None:
        raise KeyError(f"Could not resolve conservation-law metric for objective={objective_name!r}.")
    return float(value)


class NullLogger:
    def log(self, values: dict[str, Any], step: int | None = None) -> None:
        del values, step

    def log_image(self, key: str, image_path: str | Path, step: int | None = None, caption: str | None = None) -> None:
        del key, image_path, step, caption

    def log_directory_files(
        self,
        directory: str | Path,
        *,
        artifact_name: str,
        artifact_type: str = "simulation_outputs",
        exclude_dirs: list[str] | None = None,
        exclude_files: list[str] | None = None,
    ) -> None:
        del directory, artifact_name, artifact_type, exclude_dirs, exclude_files

    def finish(self) -> None:
        return


def create_rank_aware_logger(*, cfg: Any, run_id: str, output_dir: str, enabled: bool):
    if enabled:
        return WandbLogger(cfg=cfg, run_id=run_id, output_dir=output_dir)
    return NullLogger()
