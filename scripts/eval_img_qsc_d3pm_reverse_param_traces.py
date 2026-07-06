from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from experiments.common.io import ensure_dir, save_json
from experiments.common.repro import configure_environment, get_best_device, set_global_seed
from scripts.eval_img_generation_checkpoints import (
    _ksc_lambda_from_t,
    _ksc_t_from_lambda,
    build_image_model,
    load_model_state_dict,
    qsc_d3pm_posterior_logits_from_x0_posterior,
    sample_from_logits,
    unwrap_wandb_config_values,
)
from scripts.img_common import load_img_benchmark
from src.channels import log_ksc_kernel
from src.img_data import save_image_grid
from src.lm_distributed import unwrap_model


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate traces for the D3PM reverse-parameterization sampler (QSC).")
    parser.add_argument("--config", required=True, help="Path to a config.yaml (from a training run output dir).")
    parser.add_argument("--checkpoint", required=True, help="Path to a model checkpoint (.pt).")
    parser.add_argument("--output-dir", required=True, help="Output directory for trace artifacts.")
    parser.add_argument("--count", type=int, default=5, help="Number of images to trace (columns).")
    parser.add_argument("--num-diff-steps", type=int, default=100, help="Total reverse steps in the sampler.")
    parser.add_argument("--trace-points", type=int, default=10, help="Number of intermediate points (excludes endpoints).")
    parser.add_argument(
        "--schedule",
        choices=("uniform_t", "cosine", "equal_info", "derivative"),
        default="equal_info",
        help="Schedule over cumulative t̄. derivative requires --gexit-curve-csv.",
    )
    parser.add_argument(
        "--gexit-curve-csv",
        default=None,
        help="CSV with columns t and qsc_gexit_nats/qsc_gexit; required for schedule=derivative.",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Sampling seed.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top-k", type=int, default=0, help="Optional top-k truncation; 0 disables.")
    parser.add_argument("--device", default="auto", help="Torch device, or auto.")
    return parser


def _qsc_mutual_information_nats(*, t: torch.Tensor, alphabet_size: int, eps: float = 1e-12) -> torch.Tensor:
    k = float(alphabet_size)
    t_safe = torch.clamp(torch.as_tensor(t, dtype=torch.float64), min=eps, max=1.0 - eps)
    h_cond = -(t_safe * torch.log(t_safe) + (1.0 - t_safe) * torch.log((1.0 - t_safe) / (k - 1.0)))
    return math.log(k) - h_cond


def _equal_info_t_values(*, t_min: float, t_max: float, steps: int, alphabet_size: int) -> torch.Tensor:
    grid = torch.linspace(float(t_min), float(t_max), steps=8192, dtype=torch.float64)
    info = _qsc_mutual_information_nats(t=grid, alphabet_size=int(alphabet_size))
    info_min = float(info.min().item())
    info_max = float(info.max().item())
    target = torch.linspace(info_min, info_max, steps=int(steps) + 1, dtype=torch.float64)
    idx = torch.searchsorted(info, target, right=False)
    idx = torch.clamp(idx, 1, int(info.numel()) - 1)
    info0 = info[idx - 1]
    info1 = info[idx]
    t0 = grid[idx - 1]
    t1 = grid[idx]
    alpha = (target - info0) / torch.clamp(info1 - info0, min=1e-18)
    t_values = t0 + alpha * (t1 - t0)
    return t_values.to(dtype=torch.float32)


def _read_qsc_gexit_curve_csv(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    x_values: list[float] = []
    y_values: list[float] = []
    y_columns = ("qsc_gexit_nats", "qsc_gexit", "bsc_gexit", "bsc_gexit_nats", "gexit", "gexit_nats")
    with Path(path).expanduser().open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Curve CSV {path} does not have a header.")
        if "t" not in reader.fieldnames:
            raise ValueError(f"Curve CSV {path} must contain a t column.")
        y_column = next((name for name in y_columns if name in reader.fieldnames), None)
        if y_column is None:
            raise ValueError(f"Curve CSV {path} must contain one of {y_columns}.")
        for row in reader:
            try:
                x_values.append(float(row["t"]))
                y_values.append(float(row[y_column]))
            except (TypeError, ValueError):
                continue
    if len(x_values) < 2:
        raise ValueError(f"Curve CSV {path} must contain at least two finite curve points.")
    return torch.tensor(x_values, dtype=torch.float64), torch.tensor(y_values, dtype=torch.float64)


def _fill_nonfinite_with_nearest(y: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(y)
    if not bool(finite.any().item()):
        raise ValueError("gEXIT curve has no finite values.")
    if bool(finite.all().item()):
        return y
    indices = torch.arange(int(y.numel()), dtype=torch.long)
    finite_indices = indices[finite]
    filled = y.clone()
    for idx in indices[~finite].tolist():
        nearest = finite_indices[torch.argmin((finite_indices - int(idx)).abs())]
        filled[int(idx)] = y[int(nearest)]
    return filled


def _derivative_t_values_from_gexit_curve(
    *,
    curve_t: torch.Tensor,
    gexit_curve: torch.Tensor,
    t_min: float,
    t_max: float,
    steps: int,
) -> torch.Tensor:
    x = torch.as_tensor(curve_t, dtype=torch.float64)
    y = _fill_nonfinite_with_nearest(torch.as_tensor(gexit_curve, dtype=torch.float64))
    order = torch.argsort(x)
    x = x[order]
    y = y[order]
    valid = torch.isfinite(x) & torch.isfinite(y)
    x = x[valid]
    y = y[valid]
    keep_range = (x >= float(t_min)) & (x <= float(t_max))
    x = x[keep_range]
    y = y[keep_range]
    if int(x.numel()) < 2:
        raise ValueError("Derivative schedule needs at least two finite gEXIT points inside [t_min, t_max].")
    x[0] = float(t_min)
    x[-1] = float(t_max)
    dx = torch.clamp(x[1:] - x[:-1], min=0.0)
    slope_abs = torch.where(dx > 0.0, (y[1:] - y[:-1]).abs() / torch.clamp(dx, min=1e-18), torch.zeros_like(dx))
    segment_density = torch.sqrt(torch.nan_to_num(slope_abs, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0))
    segment_area = segment_density * dx
    cumulative = torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum(segment_area, dim=0)])
    total = float(cumulative[-1].item())
    if not math.isfinite(total) or total <= 1e-18:
        return torch.linspace(float(t_min), float(t_max), steps=int(steps) + 1, dtype=torch.float32)
    keep = torch.ones_like(cumulative, dtype=torch.bool)
    keep[1:] = cumulative[1:] > cumulative[:-1]
    cumulative = cumulative[keep] / total
    x = x[keep]
    targets = torch.linspace(0.0, 1.0, steps=int(steps) + 1, dtype=torch.float64)
    idx = torch.searchsorted(cumulative, targets, right=False).clamp(max=int(cumulative.numel()) - 1)
    left = torch.clamp(idx - 1, min=0)
    right = idx
    c0, c1 = cumulative[left], cumulative[right]
    x0, x1 = x[left], x[right]
    alpha = torch.where((c1 - c0).abs() > 1e-18, (targets - c0) / (c1 - c0), torch.zeros_like(targets))
    t_values = x0 + alpha * (x1 - x0)
    t_values[0] = float(t_min)
    t_values[-1] = float(t_max)
    return t_values.to(dtype=torch.float32)


def _make_tbar_schedule(
    *,
    t_min: float,
    t_max: float,
    steps: int,
    schedule: str,
    alphabet_size: int,
    gexit_curve_csv: str | Path | None = None,
) -> torch.Tensor:
    # We want t̄ to span clean->uniform (t̄=1 -> t̄=1/K) as the forward/cumulative schedule.
    if schedule == "uniform_t":
        t_inc = torch.linspace(float(t_min), float(t_max), steps=int(steps) + 1, dtype=torch.float32)
    elif schedule == "cosine":
        s = 0.008
        u = torch.linspace(0.0, 1.0, steps=int(steps) + 1, dtype=torch.float64)
        alpha_bar = torch.cos(((u + s) / (1.0 + s)) * math.pi / 2.0) ** 2
        alpha_bar = alpha_bar / torch.clamp(alpha_bar[0], min=1e-18)
        lam_bar = torch.clamp(alpha_bar, min=0.0, max=1.0)
        t_forward = (1.0 + (float(alphabet_size) - 1.0) * lam_bar) / float(alphabet_size)
        return t_forward.to(dtype=torch.float32)
    elif schedule == "equal_info":
        t_inc = _equal_info_t_values(t_min=t_min, t_max=t_max, steps=int(steps), alphabet_size=int(alphabet_size))
    elif schedule == "derivative":
        if gexit_curve_csv is None:
            raise ValueError("schedule='derivative' requires --gexit-curve-csv.")
        curve_t, gexit_curve = _read_qsc_gexit_curve_csv(gexit_curve_csv)
        t_inc = _derivative_t_values_from_gexit_curve(
            curve_t=curve_t,
            gexit_curve=gexit_curve,
            t_min=t_min,
            t_max=t_max,
            steps=int(steps),
        )
    else:
        raise ValueError(f"Unsupported schedule={schedule!r}.")
    return torch.flip(t_inc, dims=[0])  # clean->uniform


@torch.no_grad()
def sample_trace(
    model: torch.nn.Module,
    benchmark: Any,
    *,
    tbar: torch.Tensor,
    temperature: float,
    top_k: int,
    device: torch.device,
    generator: torch.Generator,
    count: int,
    trace_points: int,
) -> tuple[list[int], list[float], list[torch.Tensor]]:
    base_model = unwrap_model(model)
    K = int(base_model.vocab_size)
    T = int(tbar.shape[0]) - 1

    lam_bar = _ksc_lambda_from_t(tbar, K).clamp(min=0.0, max=1.0)
    lam_step = torch.zeros(T + 1, dtype=torch.float32)
    for t_idx in range(1, T + 1):
        denom = float(lam_bar[t_idx - 1].item())
        lam_step[t_idx] = 0.0 if denom <= 0.0 else float(lam_bar[t_idx].item()) / denom
    t_step = _ksc_t_from_lambda(lam_step, K).clamp(min=1.0 / float(K), max=1.0)

    # Record at endpoints + `trace_points` intermediates.
    n_frames = max(2, int(trace_points) + 2)
    record = torch.linspace(float(T), 0.0, steps=n_frames).round().to(dtype=torch.long).tolist()
    record_indices = sorted({int(idx) for idx in record}, reverse=True)
    record_set = set(record_indices)

    frames: list[torch.Tensor] = []
    frame_t_indices: list[int] = []
    frame_t_values: list[float] = []

    # Start at x_T ~ Uniform (t̄_T = 1/K).
    if generator is None or torch.device(generator.device).type == device.type:
        x_t = torch.randint(K, (int(count), benchmark.context_length), generator=generator, dtype=torch.long, device=device)
    else:
        x_t = torch.randint(K, (int(count), benchmark.context_length), generator=generator, dtype=torch.long).to(device=device)
    if T in record_set:
        frames.append(benchmark.decode_token_sequences(x_t.detach().cpu()).clamp(0.0, 1.0))
        frame_t_indices.append(T)
        frame_t_values.append(float(tbar[T].item()))

    for t_idx in range(T, 0, -1):
        t_cur = float(tbar[t_idx].item())
        t_prev = float(tbar[t_idx - 1].item())
        t_step_val = float(t_step[t_idx].item())

        t_cur_batch = torch.full((int(count),), t_cur, dtype=torch.float32, device=device)
        tilt = log_ksc_kernel(x_t, t_cur_batch, K)
        outputs = model.forward_outputs(x_t, timesteps=t_cur_batch, channel_tilt=tilt) if hasattr(model, "forward_outputs") else None
        logits = outputs.logits if outputs is not None else model(x_t, timesteps=t_cur_batch) + tilt
        x0_guess = torch.softmax(logits, dim=-1)

        logits_prev = qsc_d3pm_posterior_logits_from_x0_posterior(
            p_x0=x0_guess,
            x_t=x_t,
            t_cur=t_cur,
            t_prev=t_prev,
            t_step=t_step_val,
            alphabet_size=K,
        )
        x_t = sample_from_logits(logits_prev, temperature=temperature, top_k=top_k, generator=generator)

        t_prev_idx = t_idx - 1
        if t_prev_idx in record_set:
            frames.append(benchmark.decode_token_sequences(x_t.detach().cpu()).clamp(0.0, 1.0))
            frame_t_indices.append(int(t_prev_idx))
            frame_t_values.append(float(tbar[t_prev_idx].item()))

    return frame_t_indices, frame_t_values, frames


def main(argv: list[str] | None = None) -> int:
    configure_environment()
    argv = sys.argv[1:] if argv is None else argv
    args = build_arg_parser().parse_args(argv)

    cfg_path = Path(args.config).expanduser().resolve()
    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    output_dir = ensure_dir(Path(args.output_dir).expanduser().resolve())

    cfg = unwrap_wandb_config_values(OmegaConf.load(cfg_path))
    if str(getattr(cfg, "kind", "")) != "img_qsc":
        raise ValueError(f"Expected cfg.kind='img_qsc', got {getattr(cfg, 'kind', None)!r}.")

    device = get_best_device(str(args.device))
    set_global_seed(int(args.seed), deterministic=True)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed))

    benchmark = load_img_benchmark(cfg.benchmark, project_root=APP_ROOT.parent)
    model = build_image_model(cfg, vocab_size=int(benchmark.vocab_size), context_length=int(benchmark.context_length)).to(device)
    model.load_state_dict(load_model_state_dict(ckpt_path), strict=True)
    model.eval()

    base_model = unwrap_model(model)
    # Explicitly span clean->uniform as requested.
    t_min = 1.0 / float(base_model.vocab_size)
    t_max = 1.0
    tbar = _make_tbar_schedule(
        t_min=t_min,
        t_max=t_max,
        steps=int(args.num_diff_steps),
        schedule=str(args.schedule),
        alphabet_size=int(base_model.vocab_size),
        gexit_curve_csv=args.gexit_curve_csv,
    ).to(device=device)

    t_indices, t_values, frames = sample_trace(
        model,
        benchmark,
        tbar=tbar,
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        device=device,
        generator=generator,
        count=int(args.count),
        trace_points=int(args.trace_points),
    )

    rows = []
    for idx, t_val, frame in zip(t_indices, t_values, frames, strict=True):
        rows.append((f"x_t={idx:04d}_tbar={t_val:.6f}", frame))

    save_image_grid(output_dir / "qsc_d3pm_reverse_param_trace.png", rows=rows)
    save_json(
        output_dir / "trace_metadata.json",
        {
            "sampler": "qsc_d3pm_reverse_param",
            "config_path": str(cfg_path),
            "checkpoint_path": str(ckpt_path),
            "count": int(args.count),
            "num_diff_steps": int(args.num_diff_steps),
            "trace_points": int(args.trace_points),
            "schedule": str(args.schedule),
            "gexit_curve_csv": str(Path(args.gexit_curve_csv).expanduser()) if args.gexit_curve_csv else None,
            "t_indices": [int(x) for x in t_indices],
            "t_values": [float(x) for x in t_values],
            "tbar_schedule": [float(x) for x in tbar.detach().cpu().tolist()],
            "seed": int(args.seed),
            "temperature": float(args.temperature),
            "top_k": int(args.top_k),
            "device": str(device),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
