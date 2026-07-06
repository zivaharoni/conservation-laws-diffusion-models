from __future__ import annotations

from pathlib import Path
import math
from typing import Any

import torch

from scripts.lm_common import (
    NullLogger,
    compose_lm_config,
    count_parameters,
    create_rank_aware_logger,
    eval_generate_curves,
    eval_log_conservation_integral,
    find_app_root,
    find_repo_root,
    lm_embedding_norm,
    load_checkpoint_if_requested,
    load_history_prefix,
    load_initial_weights_if_requested,
    maybe_apply_lm_lr_schedule,
    maybe_reset_peak_memory,
    prefixed_scalar_metrics,
    resolve_test_split_name,
    resolve_valid_split_name,
    resolve_output_dir,
    resolve_project_path,
    runtime_metrics,
    save_training_checkpoint,
)
from src.img_data import ImageTokenBenchmark, load_cifar10_image_benchmark, save_image_grid


def compose_img_config(*, project_root: Path, argv: list[str]):
    return compose_lm_config(project_root=project_root, argv=argv)


def conservation_law_nats(*, objective: str, metrics: dict[str, Any]) -> float:
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


def image_patch_dim(benchmark: ImageTokenBenchmark) -> int:
    return int(getattr(benchmark, "dimensions_per_token", int(benchmark.patch_size) * int(benchmark.patch_size) * 3))


def nats_per_token_to_bpd(*, benchmark: ImageTokenBenchmark, nats_per_token: float) -> float:
    return float(nats_per_token) / (float(image_patch_dim(benchmark)) * math.log(2.0))


def conservation_law_bpd(*, benchmark: ImageTokenBenchmark, objective: str, metrics: dict[str, Any]) -> float:
    return nats_per_token_to_bpd(
        benchmark=benchmark,
        nats_per_token=conservation_law_nats(objective=objective, metrics=metrics),
    )


def attach_bpd_to_split_metrics(
    *,
    benchmark: ImageTokenBenchmark,
    objective: str,
    split_metrics: dict[str, dict[str, Any]],
) -> None:
    for values in split_metrics.values():
        nats_per_token = values.get("nats_per_token")
        if nats_per_token is not None:
            values["bpd"] = nats_per_token_to_bpd(benchmark=benchmark, nats_per_token=float(nats_per_token))
        try:
            values["conservation_law_bpd"] = conservation_law_bpd(
                benchmark=benchmark,
                objective=objective,
                metrics=values,
            )
        except KeyError:
            continue


def load_img_benchmark(cfg, *, project_root: Path) -> ImageTokenBenchmark:
    benchmark_name = str(cfg.name).lower()
    if benchmark_name not in {"cifar10_image_tokens", "cifar10_rgb8_tokens"}:
        raise ValueError(f"Unsupported image benchmark {cfg.name!r}.")
    return load_cifar10_image_benchmark(cfg, project_root=project_root)


@torch.no_grad()
def save_discrete_reconstruction_panel(
    benchmark: ImageTokenBenchmark,
    *,
    clean_tokens: torch.Tensor,
    observed_tokens: torch.Tensor,
    predicted_tokens: torch.Tensor,
    output_path: Path,
    mask_token_id: int | None = None,
    clean_images: torch.Tensor | None = None,
) -> None:
    clean_images_t = (
        torch.as_tensor(clean_images, dtype=torch.float32, device="cpu")
        if clean_images is not None
        else benchmark.decode_token_sequences(clean_tokens, mask_token_id=None)
    )
    observed_images = benchmark.decode_token_sequences(observed_tokens, mask_token_id=mask_token_id)
    predicted_images = benchmark.decode_token_sequences(predicted_tokens, mask_token_id=None)
    save_image_grid(output_path, rows=[("clean", clean_images_t), ("observed", observed_images), ("pred", predicted_images)])
