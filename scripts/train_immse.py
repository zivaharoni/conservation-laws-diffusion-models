from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from collections.abc import Mapping

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "gexit-diffusion-matplotlib"))

import torch
from omegaconf import DictConfig, OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common.config import compose_config, resolve_config_name_and_overrides
from experiments.common.evaluation import evaluate_model, save_mmse_curve_plot
from experiments.common.experiment_registry import require_experiment_kind
from experiments.common.io import deterministic_output_dir, save_config, save_csv, save_json
from experiments.common.repro import configure_environment, get_best_device, make_run_id, set_global_seed
from experiments.common.runtime import (
    build_data_spec,
    build_model,
    loss_estimator,
    loss_name,
    loss_topk,
    print_reference_summary,
    validate_config,
)
from experiments.common.tikz_export import export_curve_table
from experiments.common.training_callbacks import build_training_callback
from experiments.common.wandb_logger import WandbLogger
from src.trainer import TrainingRequest, train_request


def _resolve_grad_clip_config(cfg: DictConfig) -> tuple[float, float, float, float]:
    grad_clip_cfg = cfg.train.grad_clip
    if OmegaConf.is_dict(grad_clip_cfg):
        return (
            float(grad_clip_cfg.embedding),
            float(grad_clip_cfg.get("attention", grad_clip_cfg.other)),
            float(grad_clip_cfg.alphabet_proj),
            float(grad_clip_cfg.other),
        )
    grad_clip = float(grad_clip_cfg)
    return grad_clip, grad_clip, grad_clip, grad_clip


def _resolve_init_weights_path(path_value: str | os.PathLike[str], *, project_root: Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return path


def _extract_state_dict(payload: object) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model_state_dict"):
            state_dict = payload.get(key)
            if isinstance(state_dict, Mapping) and all(isinstance(name, str) for name in state_dict):
                return state_dict
        if all(isinstance(name, str) for name in payload) and all(
            isinstance(value, (torch.Tensor, torch.nn.Parameter)) for value in payload.values()
        ):
            return payload
    raise TypeError("Expected a checkpoint containing a model state dict.")


def _maybe_load_initial_weights(
    model: torch.nn.Module,
    cfg: DictConfig,
    *,
    project_root: Path,
    verbose: bool = False,
) -> Path | None:
    init_weights_path = cfg.train.get("init_weights_path")
    if init_weights_path in (None, ""):
        return None

    checkpoint_path = _resolve_init_weights_path(str(init_weights_path), project_root=project_root)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Initial weights file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)
    try:
        model.load_state_dict(state_dict, strict=bool(cfg.train.get("init_weights_strict", True)))
    except RuntimeError as exc:
        raise RuntimeError(f"Failed to load initial weights from {checkpoint_path}: {exc}") from exc

    if verbose:
        print(f"[setup] initialized model weights from {checkpoint_path}")
    return checkpoint_path


def resolve_training_batching(cfg: DictConfig) -> tuple[int, int]:
    configured_batch_size = int(cfg.train.batch_size)
    configured_grad_accum = int(cfg.train.grad_accum_steps)
    microbatch_size_cfg = cfg.train.get("microbatch_size", None)
    if microbatch_size_cfg in (None, ""):
        return configured_batch_size, configured_grad_accum
    microbatch_size = int(microbatch_size_cfg)
    if microbatch_size <= 0:
        raise ValueError(f"train.microbatch_size must be positive when set, got {microbatch_size}.")
    if microbatch_size > configured_batch_size:
        raise ValueError(
            f"train.microbatch_size={microbatch_size} cannot exceed train.batch_size={configured_batch_size}."
        )
    if configured_batch_size % microbatch_size != 0:
        raise ValueError(
            f"train.batch_size={configured_batch_size} must be divisible by train.microbatch_size={microbatch_size}."
        )
    return microbatch_size, configured_grad_accum * (configured_batch_size // microbatch_size)


def run_experiment(
    cfg: DictConfig,
    device: torch.device,
    output_dir: Path,
    data_spec,
    project_root: Path,
    logger: WandbLogger | None = None,
    verbose: bool = False,
) -> tuple[torch.nn.Module, object, int, object, dict, dict]:
    require_experiment_kind(cfg, "immse")
    model = build_model(cfg, data_spec)
    _maybe_load_initial_weights(model, cfg, project_root=project_root, verbose=verbose)
    embed_dim = int(cfg.model.embed_dim)
    train_batch_size, train_grad_accum_steps = resolve_training_batching(cfg)
    if verbose and train_batch_size != int(cfg.train.batch_size):
        print(
            "[setup] using internal microbatching: "
            f"batch_size={cfg.train.batch_size} microbatch_size={train_batch_size} "
            f"effective_grad_accum_steps={train_grad_accum_steps}"
        )
    grad_clip_embedding, grad_clip_attention, grad_clip_alphabet_proj, grad_clip_other = _resolve_grad_clip_config(cfg)
    stop_callback = build_training_callback(
        cfg=cfg,
        device=device,
        embed_dim=embed_dim,
        data_spec=data_spec,
        output_dir=output_dir,
        logger=logger,
        verbose=verbose,
    )
    history = train_request(
        TrainingRequest(
            model=model,
            data_spec=data_spec,
            device=device,
            n_steps=cfg.train.n_steps,
            lr=cfg.train.lr,
            grad_accum_steps=train_grad_accum_steps,
            grad_clip_embedding=grad_clip_embedding,
            grad_clip_attention=grad_clip_attention,
            grad_clip_alphabet_proj=grad_clip_alphabet_proj,
            grad_clip_other=grad_clip_other,
            batch_size=train_batch_size,
            snr_min=float(cfg.snr_min),
            snr_max=float(cfg.snr_max),
            loss_name=loss_name(cfg),
            label_smoothing=cfg.loss.label_smoothing,
            estimator=loss_estimator(cfg),
            topk=loss_topk(cfg),
            weight_loss_by_snr=bool(cfg.loss.get("weight_loss_by_snr", True)),
            logger=logger,
            verbose=verbose,
            log_every=cfg.logging.log_every,
            stop_callback=stop_callback,
            seed=cfg.seed + 17,
        )
    )
    snr_grid, curves, eval_info = evaluate_model(cfg, model, device, embed_dim, data_spec)
    return model, history, embed_dim, snr_grid, curves, eval_info


def finalize_experiment(
    cfg: DictConfig,
    *,
    run_id: str,
    output_dir: Path,
    device: torch.device,
    data_spec,
    logger: WandbLogger | None,
    model: torch.nn.Module,
    history,
    embed_dim: int,
    snr_grid,
    curves,
    eval_info: dict,
) -> dict[str, object]:
    logger.log(
        {
            "train/final_loss": history.loss[-1],
            "train/final_mse": history.mse[-1],
            "train/final_divergence": history.divergence[-1],
            "train/final_nll": history.nll[-1],
            "train/final_nll_ext": history.nll_ext[-1],
        }
    )

    metrics = {
        "run_id": run_id,
        "seed": int(cfg.seed),
        "device": str(device),
        **{key: value for key, value in eval_info.items() if key not in {"title", "reference_label"}},
    }
    logger.log({"eval": metrics})

    curve_rows = [
        {
            "snr": float(snr),
            "true_mmse": float(true_mmse),
            "model_mmse": float(model_mmse),
            "true_half_mmse": float(true_mmse / 2.0),
            "model_half_mmse": float(model_mmse / 2.0),
        }
        for snr, true_mmse, model_mmse in zip(snr_grid, curves["true_mmse"], curves["model_mmse"])
    ]
    save_csv(output_dir / cfg.paths.history_filename, history.rows())
    save_csv(output_dir / cfg.paths.curve_filename, curve_rows)
    save_json(output_dir / cfg.paths.metrics_filename, metrics)
    export_curve_table(
        output_dir / cfg.paths.tikz_curve_filename,
        [(row["snr"], row["model_half_mmse"]) for row in curve_rows],
        x_name="snr",
        y_name="half_mmse",
    )
    save_mmse_curve_plot(
        output_dir / cfg.paths.curve_plot_filename,
        snr_grid=snr_grid,
        true_mmse=curves["true_mmse"],
        model_mmse=curves["model_mmse"],
        true_area=metrics["reference_area_nats"],
        model_area=metrics["model_area_nats"],
        title=eval_info["title"],
    )
    logger.log_image("eval/mmse_curve", output_dir / cfg.paths.curve_plot_filename, caption="final")
    torch.save(model.state_dict(), output_dir / cfg.paths.model_filename)
    logger.log_directory_files(
        output_dir,
        artifact_name=f"{run_id}-outputs",
        exclude_dirs=[str(cfg.paths.wandb_dir_name)],
        exclude_files=[str(cfg.paths.curve_plot_filename)],
    )
    logger.finish()

    return {
        "run_id": run_id,
        "output_dir": output_dir,
        "device": device,
        "data_spec": data_spec,
        "model": model,
        "history": history,
        "embed_dim": embed_dim,
        "snr_grid": snr_grid,
        "curves": curves,
        "eval_info": eval_info,
        "metrics": metrics,
        "curve_rows": curve_rows,
    }


def execute_experiment(
    cfg: DictConfig,
    *,
    config_name: str,
    overrides: list[str] | None = None,
    project_root: Path = PROJECT_ROOT,
    print_reference: bool = True,
) -> dict[str, object]:
    require_experiment_kind(cfg, "immse")
    configure_environment()
    overrides = [] if overrides is None else list(overrides)
    cfg.paths.results_root = str((Path(project_root) / str(cfg.paths.results_root)).resolve())
    validate_config(cfg)
    set_global_seed(cfg.seed, deterministic=cfg.deterministic)
    device = get_best_device(cfg.device)
    data_spec = build_data_spec(cfg)
    run_id = make_run_id(config_name, overrides)
    output_dir = deterministic_output_dir(project_root, cfg.paths.results_root, cfg.name, run_id)

    if cfg.verbose:
        print(f"[setup] data={data_spec.run_label}")
        print(f"[setup] run_id={run_id}")
        print(f"[setup] device={device}")
        print(f"[setup] output_dir={output_dir}")
        print("[setup] resolved config:")
        print(OmegaConf.to_yaml(cfg, resolve=True).strip())

    save_config(output_dir / cfg.paths.config_snapshot_name, cfg)

    logger = WandbLogger(cfg=cfg, run_id=run_id, output_dir=str(output_dir / cfg.paths.wandb_dir_name))

    if print_reference:
        print_reference_summary(data_spec)
    model, history, embed_dim, snr_grid, curves, eval_info = run_experiment(
        cfg,
        device,
        output_dir,
        data_spec,
        project_root=project_root,
        logger=logger,
        verbose=cfg.verbose,
    )

    if cfg.verbose:
        print("[eval] training complete, computing and saving IMMSE artifacts")

    return finalize_experiment(
        cfg,
        run_id=run_id,
        output_dir=output_dir,
        device=device,
        data_spec=data_spec,
        logger=logger,
        model=model,
        history=history,
        embed_dim=embed_dim,
        snr_grid=snr_grid,
        curves=curves,
        eval_info=eval_info,
    )


def main() -> None:
    config_name, overrides = resolve_config_name_and_overrides(sys.argv[1:])
    cfg = compose_config(PROJECT_ROOT, sys.argv[1:])
    result = execute_experiment(cfg, config_name=config_name, overrides=overrides, project_root=PROJECT_ROOT)

    data_spec = result["data_spec"]
    run_id = result["run_id"]
    metrics = result["metrics"]
    eval_info = result["eval_info"]
    output_dir = result["output_dir"]
    print(f"{data_spec.run_label.upper()} run_id={run_id}")
    print(f"{eval_info['reference_label']}: {metrics['reference_entropy_nats']:.6f} {eval_info['reference_unit']}")
    stationary_entropy_nats = getattr(data_spec, "stationary_entropy_nats", None)
    if stationary_entropy_nats is not None:
        print(f"Stationary entropy:     {stationary_entropy_nats:.6f} nats")
    print(
        "Reference area:          "
        f"{metrics['reference_area_nats']:.6f} nats | abs={metrics['reference_abs_error']:.6f} rel={metrics['reference_rel_error']:.2%}"
    )
    print(
        "Learned-model area:      "
        f"{metrics['model_area_nats']:.6f} nats | abs={metrics['model_abs_error']:.6f} rel={metrics['model_rel_error']:.2%}"
    )
    print(f"Saved outputs under:     {output_dir}")


if __name__ == "__main__":
    main()
