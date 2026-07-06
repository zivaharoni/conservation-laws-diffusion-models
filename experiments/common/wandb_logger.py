"""Small W&B wrapper used by the experiment scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten(item, next_prefix))
        return out
    return {prefix: value}


def _merge_tags(base_tags: list[str], extra_tags_env: str) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    extra_tags = [tag.strip() for tag in str(extra_tags_env).split(",") if tag.strip()]
    for tag in [*base_tags, *extra_tags]:
        if tag not in seen:
            merged.append(tag)
            seen.add(tag)
    return merged


class WandbLogger:
    def __init__(self, cfg: Any, run_id: str, output_dir: str) -> None:
        self.enabled = bool(cfg.logging.enabled)
        self.run = None
        self.print_every = None if getattr(cfg.logging, "print_every", None) is None else max(1, int(cfg.logging.print_every))
        if not self.enabled:
            return

        import wandb

        config_dict = OmegaConf.to_container(cfg, resolve=True)
        resume_mode = str(getattr(cfg.logging, "resume", "never"))
        wandb_run_id = getattr(cfg.logging, "wandb_id", None)
        run_name = str(getattr(cfg.logging, "run_id", None) or run_id)
        tags = _merge_tags(list(cfg.logging.tags) + [cfg.name], os.environ.get("WANDB_EXTRA_TAGS", ""))
        self.run = wandb.init(
            project=cfg.logging.project,
            entity=cfg.logging.entity,
            group=cfg.logging.group,
            tags=tags,
            name=run_name,
            id=None if wandb_run_id in (None, "") else str(wandb_run_id),
            resume=resume_mode,
            mode=cfg.logging.mode,
            dir=output_dir,
            config=config_dict,
            settings=wandb.Settings(start_method=None),
        )

    def log(self, values: dict[str, Any], step: int | None = None) -> None:
        if self.run is None:
            return
        payload = _flatten(values)
        self.run.log(payload, step=step)
        if _should_print_metrics(step=step, print_every=self.print_every):
            line = _format_metrics_line(payload, step=step)
            if line:
                print(line)

    def log_image(self, key: str, image_path: str | Path, step: int | None = None, caption: str | None = None) -> None:
        if self.run is None:
            return
        import wandb

        self.run.log({key: wandb.Image(str(image_path), caption=caption)}, step=step)

    def log_files(self, paths: list[str | Path], artifact_name: str, artifact_type: str = "simulation_outputs") -> None:
        if self.run is None:
            return
        del artifact_name, artifact_type
        for path in paths:
            file_path = Path(path)
            if not file_path.is_file():
                continue
            self.run.save(str(file_path), base_path=str(file_path.parent), policy="now")

    def log_directory_files(
        self,
        directory: str | Path,
        *,
        artifact_name: str,
        artifact_type: str = "simulation_outputs",
        exclude_dirs: list[str] | None = None,
        exclude_files: list[str] | None = None,
    ) -> None:
        if self.run is None:
            return
        del artifact_name, artifact_type
        root = Path(directory)
        excluded = set(exclude_dirs or [])
        excluded_files = set(exclude_files or [])
        for file_path in sorted(root.rglob("*")):
            if not file_path.is_file():
                continue
            relative = file_path.relative_to(root)
            if relative.parts and relative.parts[0] in excluded:
                continue
            if str(relative) in excluded_files or relative.name in excluded_files:
                continue
            self.run.save(str(file_path), base_path=str(root), policy="now")

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


def _format_metrics_line(payload: dict[str, Any], step: int | None = None, max_items: int = 8) -> str:
    immse_progress_metrics = [
        ("loss", payload.get("train/loss")),
        ("cons", payload.get("train/conservation_law")),
        ("mse", payload.get("train/mse")),
        ("epoch", payload.get("train/epoch_progress")),
        ("grad", payload.get("train/grad_norm")),
        ("emb", payload.get("train/embed_norm")),
        ("lr", payload.get("train/lr")),
    ]
    exit_progress_metrics = [
        ("loss", payload.get("train/loss")),
        ("cons", payload.get("train/conservation_law")),
        ("masked_ce", payload.get("train/masked_ce")),
        ("unmasked_ce", payload.get("train/unmasked_ce")),
        ("epoch", payload.get("train/epoch_progress")),
        ("grad", payload.get("train/grad_norm")),
        ("emb", payload.get("train/embed_norm")),
        ("lr", payload.get("train/lr")),
    ]
    qsc_progress_metrics = [
        ("loss", payload.get("train/loss")),
        ("cons", payload.get("train/conservation_law")),
        ("clean_ce", payload.get("train/clean_ce")),
        ("subst_ce", payload.get("train/substitute_ce")),
        ("post_mm", payload.get("train/posterior_mismatched")),
        ("epoch", payload.get("train/epoch_progress")),
        ("grad", payload.get("train/grad_norm")),
        ("emb", payload.get("train/embed_norm")),
        ("lr", payload.get("train/lr")),
    ]
    causal_ce_progress_metrics = [
        ("ce", payload.get("train/ce_nats")),
        ("ce_bits", payload.get("train/ce_bits")),
        ("excess", payload.get("train/excess_ce_nats")),
        ("epoch", payload.get("train/epoch_progress")),
        ("grad", payload.get("train/grad_norm")),
        ("lr", payload.get("train/lr")),
    ]
    has_exit_metrics = any(
        payload.get(key) is not None for key in ("train/masked_ce", "train/unmasked_ce")
    )
    has_qsc_metrics = any(
        payload.get(key) is not None for key in ("train/clean_ce", "train/substitute_ce", "train/posterior_mismatched")
    )
    has_causal_ce_metrics = any(
        payload.get(key) is not None for key in ("train/ce_nats", "train/ce_bits", "train/excess_ce_nats")
    )
    has_immse_metrics = any(
        payload.get(key) is not None for key in ("train/mse",)
    )
    if has_exit_metrics:
        progress_metrics = exit_progress_metrics
    elif has_qsc_metrics:
        progress_metrics = qsc_progress_metrics
    elif has_causal_ce_metrics:
        progress_metrics = causal_ce_progress_metrics
    elif has_immse_metrics:
        progress_metrics = immse_progress_metrics
    else:
        progress_metrics = immse_progress_metrics
    items = [f"{key}={_format_metric_value(value)}" for key, value in progress_metrics if value is not None][:max_items]
    prefix = f"[metrics] step={step}" if step is not None else "[metrics]"
    return "" if not items else f"{prefix} " + " ".join(items)


def _format_metric_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _should_print_metrics(step: int | None, print_every: int | None) -> bool:
    if step is None:
        return True
    if print_every is None:
        return True
    return step % print_every == 0
