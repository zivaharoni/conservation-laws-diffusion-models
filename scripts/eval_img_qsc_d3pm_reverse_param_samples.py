from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from experiments.common.io import ensure_dir, save_json
from experiments.common.repro import configure_environment, get_best_device, set_global_seed
from scripts.eval_img_generation_checkpoints import build_image_model, load_model_state_dict, unwrap_wandb_config_values
from scripts.eval_img_qsc_d3pm_reverse_param_traces import _make_tbar_schedule, sample_trace
from scripts.img_common import load_img_benchmark
from src.lm_distributed import unwrap_model


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate QSC samples with the D3PM reverse-parameterization sampler.")
    parser.add_argument("--config", required=True, help="Path to a config.yaml.")
    parser.add_argument("--checkpoint", required=True, help="Path to a model checkpoint (.pt).")
    parser.add_argument("--output-dir", required=True, help="Directory to write sample grids.")
    parser.add_argument("--count", type=int, default=100, help="Generated image count.")
    parser.add_argument("--batch-size", type=int, default=50, help="Generation batch size.")
    parser.add_argument("--num-diff-steps", type=int, default=100, help="Reverse steps.")
    parser.add_argument("--schedule", choices=("uniform_t", "cosine", "equal_info", "derivative"), default="equal_info")
    parser.add_argument("--gexit-curve-csv", default=None, help="CSV with t and qsc_gexit columns; required for schedule=derivative.")
    parser.add_argument("--seed", type=int, default=2026, help="Sampling seed.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k truncation; 0 disables.")
    parser.add_argument("--grid-cols", type=int, default=10, help="Columns in the output grid image.")
    parser.add_argument("--device", default="auto", help="Torch device or auto.")
    return parser


def _save_tiled_grid(path: Path, images: torch.Tensor, *, cols: int) -> None:
    import matplotlib.pyplot as plt

    imgs = torch.as_tensor(images, dtype=torch.float32, device="cpu").clamp(0.0, 1.0)
    n_images = int(imgs.shape[0])
    cols = max(1, int(cols))
    rows = (n_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows), squeeze=False)
    for idx in range(rows * cols):
        ax = axes[idx // cols][idx % cols]
        ax.set_xticks([])
        ax.set_yticks([])
        if idx < n_images:
            ax.imshow(imgs[idx].numpy())
        else:
            ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


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
    tbar = _make_tbar_schedule(
        t_min=1.0 / float(base_model.vocab_size),
        t_max=1.0,
        steps=int(args.num_diff_steps),
        schedule=str(args.schedule),
        alphabet_size=int(base_model.vocab_size),
        gexit_curve_csv=args.gexit_curve_csv,
    ).to(device=device)

    chunks: list[torch.Tensor] = []
    generated = 0
    while generated < int(args.count):
        current_batch = min(int(args.batch_size), int(args.count) - generated)
        _indices, _t_values, frames = sample_trace(
            model,
            benchmark,
            tbar=tbar,
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            device=device,
            generator=generator,
            count=current_batch,
            trace_points=0,
        )
        chunks.append(frames[-1])
        generated += current_batch

    images = torch.cat(chunks, dim=0)[: int(args.count)]
    _save_tiled_grid(output_dir / "samples.png", images, cols=int(args.grid_cols))
    save_json(
        output_dir / "sample_metadata.json",
        {
            "sampler": "qsc_d3pm_reverse_param",
            "config_path": str(cfg_path),
            "checkpoint_path": str(ckpt_path),
            "count": int(args.count),
            "batch_size": int(args.batch_size),
            "num_diff_steps": int(args.num_diff_steps),
            "schedule": str(args.schedule),
            "gexit_curve_csv": str(Path(args.gexit_curve_csv).expanduser()) if args.gexit_curve_csv else None,
            "tbar_start": float(tbar[0].detach().cpu().item()),
            "tbar_end": float(tbar[-1].detach().cpu().item()),
            "seed": int(args.seed),
            "temperature": float(args.temperature),
            "top_k": int(args.top_k),
            "device": str(device),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
