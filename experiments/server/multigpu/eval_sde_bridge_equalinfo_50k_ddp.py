from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

APP_ROOT = Path("/path/to/gexit-diffusion/code")
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from experiments.common.io import save_csv, save_json
from scripts.eval_img_generation_checkpoints import build_image_model, checkpoint_step, compute_generation_metrics, sample_from_logits
from scripts.eval_cifar_suite import load_final_and_ema_state_dicts
from scripts.img_common import find_repo_root, load_img_benchmark
from src.lm_distributed import unwrap_model

RUN_DIR_DEFAULT = "/path/to/gexit-diffusion/code/results/img/immse/rgb8_pixel32_awgn_random_skip_nll_wd01_snr1e4_10000_highsnr/experiments-cifar10-rgb8-unet-immse-finetune-exit__model-name-img-unet-rgb8-continuous__model-input-skip-true__train-init-exit-checkpoint-path-null__train-init-exit-checkpoint-source-null__796e4c02fa"


def unwrap_cfg(cfg: Any) -> Any:
    container = OmegaConf.to_container(cfg, resolve=True)
    if isinstance(container, dict) and any(isinstance(value, dict) and set(value.keys()) == {"value"} for value in container.values()):
        return OmegaConf.create({key: value["value"] if isinstance(value, dict) and set(value.keys()) == {"value"} else value for key, value in container.items()})
    return cfg


def latest_checkpoint(run_dir: Path) -> Path:
    checkpoints = sorted((run_dir / "checkpoints").glob("model_step_*.pt"), key=checkpoint_step)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {run_dir / checkpoints}")
    return checkpoints[-1]


def load_curve(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    snr_values: list[float] = []
    half_mmse_values: list[float] = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            snr = float(row["snr"])
            half_mmse = float(row["half_mmse"]) if row.get("half_mmse") not in (None, "") else 0.5 * float(row["mmse"])
            if math.isfinite(snr) and math.isfinite(half_mmse) and snr > 0.0:
                snr_values.append(float(snr))
                half_mmse_values.append(max(0.0, float(half_mmse)))
    if len(snr_values) < 2:
        raise ValueError(f"Need at least two curve points in {path}")
    return torch.tensor(snr_values, dtype=torch.float64), torch.tensor(half_mmse_values, dtype=torch.float64)


def build_equal_info_snr_schedule(curve_path: Path, *, num_steps: int) -> torch.Tensor:
    snr, half_mmse = load_curve(curve_path)
    order = torch.argsort(snr)
    x = snr[order]
    y = torch.nan_to_num(half_mmse[order], nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    segment_area = 0.5 * (y[:-1] + y[1:]) * torch.clamp(x[1:] - x[:-1], min=0.0)
    cumulative = torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum(segment_area, dim=0)])
    total = float(cumulative[-1].item())
    if not math.isfinite(total) or total <= 1.0e-12:
        raise ValueError(f"Curve {curve_path} has non-positive equal-info area")
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
    schedule = (x0 + frac * (x1 - x0)).to(dtype=torch.float32)
    schedule[0] = x[0].to(dtype=torch.float32)
    schedule[-1] = x[-1].to(dtype=torch.float32)
    if not bool(torch.all(schedule[1:] >= schedule[:-1])):
        raise RuntimeError("Equal-info SNR schedule is not monotone increasing.")
    return schedule


@torch.inference_mode()
def generate_bridge_sde_shard(
    model: torch.nn.Module,
    benchmark: Any,
    schedule: torch.Tensor,
    *,
    count: int,
    batch_size: int,
    seed: int,
    rank: int,
    log_every: int,
) -> torch.Tensor:
    device = next(model.parameters()).device
    base_model = unwrap_model(model)
    generator = torch.Generator(device=device).manual_seed(int(seed) + 1_000_003 * int(rank))
    schedule = schedule.to(device=device, dtype=torch.float32)
    codebook = base_model.x_values.weight.to(device=device, dtype=torch.float32)
    outputs: list[torch.Tensor] = []
    done = 0
    start = time.perf_counter()
    while done < int(count):
        current = min(int(batch_size), int(count) - done)
        x = torch.randn((current, benchmark.context_length, int(base_model.x_values.embedding_dim)), generator=generator, device=device)
        x = x / torch.sqrt(torch.clamp(schedule[0].expand(current, benchmark.context_length, 1), min=1.0e-12))
        for step_idx in range(int(schedule.numel()) - 1):
            s = schedule[step_idx].expand(current, benchmark.context_length, 1)
            r = schedule[step_idx + 1].expand(current, benchmark.context_length, 1)
            _emb_hat, _logits_ext, logits = model(x, s)
            x0_tokens = sample_from_logits(logits, temperature=1.0, top_k=0, generator=generator)
            x0 = codebook[x0_tokens.long()]
            alpha = s / torch.clamp(r, min=1.0e-12)
            mean = x0 + alpha * (x - x0)
            sigma = torch.sqrt(torch.clamp((r - s) / torch.clamp(r * r, min=1.0e-24), min=0.0))
            noise = torch.randn(x.shape, generator=generator, device=device, dtype=x.dtype)
            x = mean + sigma * noise
        final_snr = schedule[-1].expand(current, benchmark.context_length, 1)
        _emb_hat, _logits_ext, logits = model(x, final_snr)
        tokens = sample_from_logits(logits, temperature=1.0, top_k=0, generator=generator)
        images = benchmark.decode_token_sequences(tokens.detach().cpu()).clamp(0.0, 1.0).to(dtype=torch.float16)
        outputs.append(images)
        done += current
        if int(log_every) > 0 and (done >= int(count) or done % int(log_every) == 0):
            elapsed = time.perf_counter() - start
            print(f"[rank {rank}] generated {done}/{count} shard samples rate={done / max(elapsed, 1e-9):.2f}/s elapsed={elapsed:.1f}s", flush=True)
    return torch.cat(outputs, dim=0)[: int(count)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=RUN_DIR_DEFAULT)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--curve", default=None)
    parser.add_argument("--sample-count", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--metric-batch-size", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=20260505)
    parser.add_argument("--log-every", type=int, default=480)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = int(dist.get_rank())
    world_size = int(dist.get_world_size())
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    torch.backends.cuda.matmul.allow_tf32 = True

    run_dir = Path(args.run_dir)
    checkpoint = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(run_dir)
    step = checkpoint_step(checkpoint)
    curve_path = Path(args.curve) if args.curve else run_dir / "immse_curve.csv"
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / f"eval_sde_bridge_step{step:07d}_equalinfo_snr1000_50k_train_ddp"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = unwrap_cfg(OmegaConf.load(run_dir / "config.yaml"))
    benchmark = load_img_benchmark(cfg.benchmark, project_root=find_repo_root(APP_ROOT))
    states = load_final_and_ema_state_dicts(checkpoint, required_variants=["ema"])
    model = build_image_model(cfg, vocab_size=benchmark.vocab_size, context_length=benchmark.context_length)
    model.load_state_dict(states["ema"])
    model.to(device).eval()

    schedule = build_equal_info_snr_schedule(curve_path, num_steps=int(args.num_steps))
    schedule_payload = {
        "schedule": "equal_info_snr",
        "curve_path": str(curve_path),
        "num_steps": int(args.num_steps),
        "snr_min": float(schedule[0].item()),
        "snr_max": float(schedule[-1].item()),
        "snr_values": [float(value) for value in schedule.cpu().tolist()],
    }

    base_count = int(args.sample_count) // world_size
    extra = int(args.sample_count) % world_size
    shard_count = base_count + (1 if rank < extra else 0)
    if rank == 0:
        meta = {
            "checkpoint": str(checkpoint),
            "checkpoint_step": int(step),
            "weight_variant": "ema",
            "sampler": "stochastic_gaussian_bridge_sde",
            "transition": "x_r = x0_sample + (s/r)*(x_s-x0_sample) + sqrt((r-s)/r^2)*z",
            "schedule": "equal_info_snr",
            "curve_path": str(curve_path),
            "snr_min": float(schedule[0].item()),
            "snr_max": float(schedule[-1].item()),
            "num_steps": int(args.num_steps),
            "sample_count": int(args.sample_count),
            "world_size": world_size,
            "batch_size_per_rank": int(args.batch_size),
            "split": str(args.split),
            "seed": int(args.seed),
            "started_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        save_json(output_dir / "manifest.json", meta)
        save_json(output_dir / "schedule_equal_info_snr.json", schedule_payload)
        print(json.dumps(meta, indent=2), flush=True)

    start = time.perf_counter()
    samples = generate_bridge_sde_shard(
        model,
        benchmark,
        schedule,
        count=shard_count,
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        rank=rank,
        log_every=int(args.log_every),
    )
    shard_path = output_dir / f"samples_rank{rank:02d}.pt"
    torch.save(samples, shard_path)
    print(f"[rank {rank}] wrote {shard_path} shape={tuple(samples.shape)} elapsed={time.perf_counter() - start:.1f}s", flush=True)

    del samples, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    dist.barrier()

    if rank == 0:
        metric_start = time.perf_counter()
        print("[rank 0] loading shards for metrics", flush=True)
        generated = torch.cat([torch.load(output_dir / f"samples_rank{worker_rank:02d}.pt", map_location="cpu").to(dtype=torch.float32) for worker_rank in range(world_size)], dim=0)[: int(args.sample_count)]
        real_images = benchmark.split_images(str(args.split))
        if real_images is None:
            raise ValueError(f"Benchmark split {args.split!r} has no real images for FID")
        metrics = compute_generation_metrics(
            generated,
            real_images,
            sample_count=int(args.sample_count),
            batch_size=int(args.metric_batch_size),
            device=device,
        )
        row = {
            "sampler": "stochastic_gaussian_bridge_sde",
            "checkpoint": str(checkpoint),
            "checkpoint_step": int(step),
            "weight_variant": "ema",
            "schedule": "equal_info_snr",
            "curve_path": str(curve_path),
            "snr_min": float(schedule[0].item()),
            "snr_max": float(schedule[-1].item()),
            "num_steps": int(args.num_steps),
            "sample_count": int(args.sample_count),
            "split": str(args.split),
            "world_size": world_size,
            "batch_size_per_rank": int(args.batch_size),
            "metric_batch_size": int(args.metric_batch_size),
            "seed": int(args.seed),
            "elapsed_generation_seconds": float(metric_start - start),
            "elapsed_metric_seconds": float(time.perf_counter() - metric_start),
            **metrics,
        }
        save_csv(output_dir / "metrics.csv", [row])
        save_json(output_dir / "metrics.json", row)
        print(json.dumps(row, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
