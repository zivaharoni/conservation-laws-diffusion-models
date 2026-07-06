from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def is_main_process(self) -> bool:
        return int(self.rank) == 0


def setup_distributed(requested_device: str = "auto") -> tuple[DistributedContext, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1

    device = _resolve_device(requested_device=requested_device, enabled=enabled, local_rank=local_rank)
    if enabled and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    return DistributedContext(enabled=enabled, rank=rank, world_size=world_size, local_rank=local_rank), device


def cleanup_distributed(ctx: DistributedContext) -> None:
    if ctx.enabled and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def barrier(ctx: DistributedContext) -> None:
    if ctx.enabled and dist.is_initialized():
        dist.barrier()


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def all_reduce_sum_tensor(ctx: DistributedContext, tensor: torch.Tensor) -> torch.Tensor:
    if ctx.enabled and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def all_reduce_sum_scalars(ctx: DistributedContext, values: list[float], *, device: torch.device | str) -> list[float]:
    target_device = torch.device(device)
    dtype = torch.float32 if target_device.type == "mps" else torch.float64
    tensor = torch.tensor(values, dtype=dtype, device=target_device)
    all_reduce_sum_tensor(ctx, tensor)
    return tensor.tolist()


def all_reduce_mean_scalars(ctx: DistributedContext, values: list[float], *, device: torch.device | str) -> list[float]:
    reduced = all_reduce_sum_scalars(ctx, values, device=device)
    if ctx.enabled:
        return [value / float(ctx.world_size) for value in reduced]
    return reduced


def _resolve_device(*, requested_device: str, enabled: bool, local_rank: int) -> torch.device:
    requested = str(requested_device).lower()
    if requested == "auto":
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}" if enabled else f"cuda:{torch.cuda.current_device()}")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() and not enabled:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        if requested == "cuda":
            device = torch.device(f"cuda:{local_rank}" if enabled else f"cuda:{torch.cuda.current_device()}")
        else:
            device = torch.device(requested)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device
