"""Runtime reproducibility helpers."""

from __future__ import annotations

import hashlib
import os
import random
import re

import torch


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_best_device(requested: str = "auto") -> torch.device:
    requested = str(requested).lower()
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_run_id(config_name: str, overrides: list[str]) -> str:
    parts = [_slugify(config_name)]
    for override in overrides:
        parts.append(_slugify(override))
    run_id = "__".join(part for part in parts if part)
    if len(run_id) <= 200:
        return run_id
    digest = hashlib.sha1(run_id.encode("utf-8")).hexdigest()[:10]
    return f"{run_id[:189].rstrip('_-')}__{digest}"


def configure_environment() -> None:
    os.environ.setdefault("WANDB_SILENT", "false")
    os.environ.pop("WANDB_START_METHOD", None)
    mpl_dir = os.environ.setdefault("MPLCONFIGDIR", "/tmp/gexit-diffusion-matplotlib")
    os.makedirs(mpl_dir, exist_ok=True)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", str(value)).strip("-").lower()
    return slug or "default"
