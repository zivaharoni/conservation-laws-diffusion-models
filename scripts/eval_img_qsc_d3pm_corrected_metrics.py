from __future__ import annotations

import argparse
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
from scripts.eval_img_generation_checkpoints import (
    build_image_model,
    checkpoint_step,
    compute_generation_metrics,
    load_model_state_dict,
    unwrap_wandb_config_values,
)
from scripts.eval_img_qsc_d3pm_reverse_param_samples import _save_tiled_grid
from scripts.eval_img_qsc_d3pm_reverse_param_traces import _make_tbar_schedule, sample_trace
from scripts.img_common import find_repo_root, load_img_benchmark
from src.lm_distributed import unwrap_model


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate corrected D3PM QSC sampler FID/IS for one checkpoint.")
    parser.add_argument("--config", required=True, help="Path to config.yaml.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--sample-count", type=int, default=50_000, help="Generated sample count.")
    parser.add_argument("--batch-size", type=int, default=4096, help="Initial generation batch size.")
    parser.add_argument("--min-batch-size", type=int, default=64, help="Smallest generation batch size after OOM backoff.")
    parser.add_argument("--metric-batch-size", type=int, default=512, help="Batch size for FID/IS network updates.")
    parser.add_argument("--num-diff-steps", type=int, default=1000, help="D3PM reverse diffusion steps.")
    parser.add_argument("--schedule", choices=("uniform_t", "equal_info", "cosine", "derivative"), required=True)
    parser.add_argument("--gexit-curve-csv", default=None, help="CSV with t and qsc_gexit columns; required for schedule=derivative.")
    parser.add_argument("--weight-variant", choices=("final", "ema"), default="ema")
    parser.add_argument("--split", default="test", help="Real-image split for FID.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--grid-count", type=int, default=100)
    parser.add_argument("--grid-cols", type=int, default=10)
    parser.add_argument("--device", default="auto")
    return parser


def _normalize_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {key[len("module.") :] if key.startswith("module.") else key: value for key, value in state_dict.items()}


def _load_variant_state_dict(path: str | Path, *, variant: str) -> dict[str, Any]:
    if str(variant) == "final":
        return load_model_state_dict(path)
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("ema_state_dict"), dict):
        raise KeyError(f"Checkpoint {path} does not contain ema_state_dict.")
    return _normalize_state_dict_keys(payload["ema_state_dict"])


@torch.no_grad()
def generate_corrected_d3pm_images(
    model: torch.nn.Module,
    benchmark: Any,
    cfg: Any,
    *,
    sample_count: int,
    initial_batch_size: int,
    min_batch_size: int,
    num_diff_steps: int,
    schedule_name: str,
    gexit_curve_csv: str | Path | None,
    temperature: float,
    top_k: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    base_model = unwrap_model(model)
    tbar = _make_tbar_schedule(
        t_min=1.0 / float(base_model.vocab_size),
        t_max=1.0,
        steps=int(num_diff_steps),
        schedule=str(schedule_name),
        alphabet_size=int(base_model.vocab_size),
        gexit_curve_csv=gexit_curve_csv,
    ).to(device=device)

    generated: list[torch.Tensor] = []
    total = 0
    batch_size = int(initial_batch_size)
    min_batch_size = int(min_batch_size)
    while total < int(sample_count):
        current_batch = min(batch_size, int(sample_count) - total)
        try:
            _indices, _t_values, frames = sample_trace(
                model,
                benchmark,
                tbar=tbar,
                temperature=float(temperature),
                top_k=int(top_k),
                device=device,
                generator=generator,
                count=int(current_batch),
                trace_points=0,
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or batch_size <= min_batch_size:
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            batch_size = max(min_batch_size, batch_size // 2)
            print(f"[generate] OOM; retrying with batch_size={batch_size}", flush=True)
            continue
        generated.append(frames[-1].cpu())
        total += int(current_batch)
        print(f"[generate] {total}/{int(sample_count)} samples batch_size={batch_size}", flush=True)
    return torch.cat(generated, dim=0)[: int(sample_count)].clamp(0.0, 1.0), int(batch_size)


def main(argv: list[str] | None = None) -> int:
    configure_environment()
    args = build_arg_parser().parse_args(argv)
    set_global_seed(int(args.seed), deterministic=False)
    device = get_best_device(str(args.device))
    cfg_path = Path(args.config).expanduser().resolve()
    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    output_dir = ensure_dir(Path(args.output_dir).expanduser().resolve())
    cfg = unwrap_wandb_config_values(OmegaConf.load(cfg_path))
    if str(cfg.kind) != "img_qsc":
        raise ValueError(f"Expected cfg.kind='img_qsc', got {getattr(cfg, 'kind', None)!r}.")
    benchmark = load_img_benchmark(cfg.benchmark, project_root=find_repo_root(APP_ROOT))
    real_images = benchmark.split_images(str(args.split))
    if real_images is None:
        raise ValueError(f"Benchmark split {args.split!r} does not include real images for FID.")

    model = build_image_model(cfg, vocab_size=int(benchmark.vocab_size), context_length=int(benchmark.context_length))
    model.load_state_dict(_load_variant_state_dict(ckpt_path, variant=str(args.weight_variant)), strict=True)
    model = model.to(device).eval()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed))
    start = time.perf_counter()
    generated, effective_batch_size = generate_corrected_d3pm_images(
        model,
        benchmark,
        cfg,
        sample_count=int(args.sample_count),
        initial_batch_size=int(args.batch_size),
        min_batch_size=int(args.min_batch_size),
        num_diff_steps=int(args.num_diff_steps),
        schedule_name=str(args.schedule),
        gexit_curve_csv=args.gexit_curve_csv,
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        device=device,
        generator=generator,
    )
    grid_count = min(int(args.grid_count), int(generated.shape[0]))
    grid_path = output_dir / "samples_grid.png"
    _save_tiled_grid(grid_path, generated[:grid_count], cols=int(args.grid_cols))
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
        "checkpoint_path": str(ckpt_path),
        "checkpoint_step": int(checkpoint_step(ckpt_path)),
        "weight_variant": str(args.weight_variant),
        "schedule": str(args.schedule),
        "gexit_curve_csv": str(Path(args.gexit_curve_csv).expanduser()) if args.gexit_curve_csv else None,
        "sample_count": int(args.sample_count),
        "num_diff_steps": int(args.num_diff_steps),
        "requested_batch_size": int(args.batch_size),
        "effective_batch_size": int(effective_batch_size),
        "metric_batch_size": int(args.metric_batch_size),
        "split": str(args.split),
        "seed": int(args.seed),
        "temperature": float(args.temperature),
        "top_k": int(args.top_k),
        "device": str(device),
        "elapsed_seconds": float(time.perf_counter() - start),
        "sample_grid": str(grid_path),
        **metrics,
    }
    save_csv(output_dir / "metrics.csv", [row])
    save_json(output_dir / "metrics.json", row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
