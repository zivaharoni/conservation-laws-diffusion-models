from __future__ import annotations

import argparse
import json
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
from scripts.eval_img_generation_checkpoints import build_image_model, checkpoint_step, compute_generation_metrics, load_model_state_dict, unwrap_wandb_config_values
from scripts.eval_img_qsc_d3pm_corrected_metrics import _load_variant_state_dict
from scripts.eval_img_qsc_d3pm_reverse_param_samples import _save_tiled_grid
from scripts.eval_img_qsc_d3pm_reverse_param_traces import _make_tbar_schedule, sample_trace
from scripts.img_common import find_repo_root, load_img_benchmark

RUN_DIR_DEFAULT = "/path/to/gexit-diffusion/code/results/img/qsc/cifar10_rgb8_unet_qsc/experiments-cifar10-rgb8-unet-qsc__train-batch-size-1024__train-grad-accum-steps-1__eval-batch-size-256__model-input-skip-true__train-init-weights-path-null__train-init-weights-strict-false__ae798723cc"


def latest_checkpoint(run_dir: Path) -> Path:
    checkpoints = sorted((run_dir / "checkpoints").glob("model_step_*.pt"), key=checkpoint_step)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {run_dir / checkpoints}")
    return checkpoints[-1]


@torch.no_grad()
def generate_qsc_shard(
    model: torch.nn.Module,
    benchmark: Any,
    tbar: torch.Tensor,
    *,
    count: int,
    batch_size: int,
    min_batch_size: int,
    seed: int,
    rank: int,
    schedule: str,
    temperature: float,
    top_k: int,
    log_every: int,
) -> tuple[torch.Tensor, int]:
    device = next(model.parameters()).device
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) + 1_000_003 * int(rank))
    outputs: list[torch.Tensor] = []
    done = 0
    effective_batch = int(batch_size)
    start = time.perf_counter()
    while done < int(count):
        current = min(int(effective_batch), int(count) - done)
        try:
            _indices, _t_values, frames = sample_trace(
                model,
                benchmark,
                tbar=tbar,
                temperature=float(temperature),
                top_k=int(top_k),
                device=device,
                generator=generator,
                count=int(current),
                trace_points=0,
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or effective_batch <= int(min_batch_size):
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            effective_batch = max(int(min_batch_size), int(effective_batch) // 2)
            print(f"[rank {rank}] OOM; retry batch_size={effective_batch}", flush=True)
            continue
        outputs.append(frames[-1].cpu().to(dtype=torch.float16))
        done += int(current)
        if int(log_every) > 0 and (done >= int(count) or done % int(log_every) == 0):
            elapsed = time.perf_counter() - start
            print(f"[rank {rank}] {schedule} generated {done}/{count} shard samples rate={done / max(elapsed, 1e-9):.2f}/s elapsed={elapsed:.1f}s batch={effective_batch}", flush=True)
    return torch.cat(outputs, dim=0)[: int(count)], int(effective_batch)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=RUN_DIR_DEFAULT)
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sample-count", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--min-batch-size", type=int, default=64)
    parser.add_argument("--metric-batch-size", type=int, default=512)
    parser.add_argument("--num-diff-steps", type=int, default=1000)
    parser.add_argument("--schedule", choices=("uniform_t", "equal_info", "cosine", "derivative"), required=True)
    parser.add_argument("--gexit-curve-csv", default=None, help="CSV with t and qsc_gexit columns; required for schedule=derivative.")
    parser.add_argument("--weight-variant", choices=("final", "ema"), default="final")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=5103)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--grid-count", type=int, default=100)
    parser.add_argument("--grid-cols", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=512)
    parser.add_argument("--skip-metrics", action="store_true", help="Only generate shard samples and skip FID/IS metrics.")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = int(dist.get_rank())
    world_size = int(dist.get_world_size())
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    torch.backends.cuda.matmul.allow_tf32 = True

    run_dir = Path(args.run_dir)
    cfg_path = Path(args.config) if args.config else run_dir / "config.yaml"
    ckpt = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(run_dir)
    step = checkpoint_step(ckpt)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "eval_fid_is_d3pm_corrected_50k_multigpu" / f"model_step_{step:07d}__{args.weight_variant}__{args.schedule}__steps-{int(args.num_diff_steps)}__n-{int(args.sample_count)}__seed-{int(args.seed)}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = unwrap_wandb_config_values(OmegaConf.load(cfg_path))
    benchmark = load_img_benchmark(cfg.benchmark, project_root=find_repo_root(APP_ROOT))
    model = build_image_model(cfg, vocab_size=int(benchmark.vocab_size), context_length=int(benchmark.context_length))
    model.load_state_dict(_load_variant_state_dict(ckpt, variant=str(args.weight_variant)), strict=True)
    model.to(device).eval()

    tbar = _make_tbar_schedule(
        t_min=1.0 / float(benchmark.vocab_size),
        t_max=1.0,
        steps=int(args.num_diff_steps),
        schedule=str(args.schedule),
        alphabet_size=int(benchmark.vocab_size),
        gexit_curve_csv=args.gexit_curve_csv,
    ).to(device=device)

    base = int(args.sample_count) // world_size
    extra = int(args.sample_count) % world_size
    shard_count = base + (1 if rank < extra else 0)

    if rank == 0:
        manifest = {
            "sampler": "qsc_d3pm_reverse_param_corrected",
            "config_path": str(cfg_path),
            "checkpoint_path": str(ckpt),
            "checkpoint_step": int(step),
            "weight_variant": str(args.weight_variant),
            "schedule": str(args.schedule),
            "gexit_curve_csv": str(Path(args.gexit_curve_csv).expanduser()) if args.gexit_curve_csv else None,
            "sample_count": int(args.sample_count),
            "num_diff_steps": int(args.num_diff_steps),
            "world_size": int(world_size),
            "batch_size_per_rank": int(args.batch_size),
            "min_batch_size": int(args.min_batch_size),
            "metric_batch_size": int(args.metric_batch_size),
            "split": str(args.split),
            "seed": int(args.seed),
            "temperature": float(args.temperature),
            "top_k": int(args.top_k),
            "tbar_start": float(tbar[0].detach().cpu().item()),
            "tbar_end": float(tbar[-1].detach().cpu().item()),
            "started_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        save_json(output_dir / "manifest.json", manifest)
        print(json.dumps(manifest, indent=2), flush=True)

    start = time.perf_counter()
    samples, effective_batch = generate_qsc_shard(
        model,
        benchmark,
        tbar,
        count=shard_count,
        batch_size=int(args.batch_size),
        min_batch_size=int(args.min_batch_size),
        seed=int(args.seed),
        rank=rank,
        schedule=str(args.schedule),
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        log_every=int(args.log_every),
    )
    shard_path = output_dir / f"samples_rank{rank:02d}.pt"
    torch.save(samples, shard_path)
    print(f"[rank {rank}] wrote {shard_path} shape={tuple(samples.shape)} elapsed={time.perf_counter()-start:.1f}s", flush=True)

    del samples, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    dist.barrier()
    dist.destroy_process_group()

    if rank == 0:
        metric_start = time.perf_counter()
        generated = torch.cat([torch.load(output_dir / f"samples_rank{r:02d}.pt", map_location="cpu").to(dtype=torch.float32) for r in range(world_size)], dim=0)[: int(args.sample_count)]
        grid_count = min(int(args.grid_count), int(generated.shape[0]))
        grid_path = output_dir / "samples_grid.png"
        _save_tiled_grid(grid_path, generated[:grid_count], cols=int(args.grid_cols))
        metrics = {}
        if not bool(args.skip_metrics):
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
            "sampler": "qsc_d3pm_reverse_param_corrected",
            "config_path": str(cfg_path),
            "checkpoint_path": str(ckpt),
            "checkpoint_step": int(step),
            "weight_variant": str(args.weight_variant),
            "schedule": str(args.schedule),
            "gexit_curve_csv": str(Path(args.gexit_curve_csv).expanduser()) if args.gexit_curve_csv else None,
            "sample_count": int(args.sample_count),
            "num_diff_steps": int(args.num_diff_steps),
            "world_size": int(world_size),
            "requested_batch_size_per_rank": int(args.batch_size),
            "effective_batch_size_rank0": int(effective_batch),
            "metric_batch_size": int(args.metric_batch_size),
            "split": str(args.split),
            "seed": int(args.seed),
            "temperature": float(args.temperature),
            "top_k": int(args.top_k),
            "elapsed_generation_seconds": float(metric_start - start),
            "elapsed_metric_seconds": float(time.perf_counter() - metric_start),
            "sample_grid": str(grid_path),
            "metrics_skipped": bool(args.skip_metrics),
            **metrics,
        }
        save_csv(output_dir / "metrics.csv", [row])
        save_json(output_dir / "metrics.json", row)
        print(json.dumps(row, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
