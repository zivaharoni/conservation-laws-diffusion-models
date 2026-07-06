from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from experiments.common.io import save_csv, save_json
from experiments.common.repro import configure_environment
from scripts.lm_common import (
    conservation_law_nats,
    count_parameters,
    create_rank_aware_logger,
    find_repo_root,
    prefixed_scalar_metrics,
)
from src.lm_data import load_lm_benchmark
from src.lm_models import CausalTransformerLM, ContinuousDiffusionTransformerLM, MaskedTransformerLM
from src.lm_objectives import (
    evaluate_causal_nats,
    evaluate_exit_curve,
    evaluate_exit_objective_metrics,
    evaluate_immse_curve,
    evaluate_immse_objective_metrics,
    evaluate_qsc_curve,
    evaluate_qsc_objective_metrics,
    save_exit_curve_plot,
    save_immse_curve_plot,
    save_qsc_curve_plot,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a saved LM checkpoint and write final artifacts.")
    parser.add_argument("--checkpoint", required=True, help="Path to a saved training checkpoint.")
    parser.add_argument("--config", default=None, help="Optional path to the saved config.yaml for the run.")
    parser.add_argument("--split", default="test", help="Evaluation split: val or test.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional eval batch size override.")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional max batches for quick evaluation.")
    parser.add_argument("--output-json", default=None, help="Optional path to save the full JSON payload.")
    parser.add_argument("--wandb-group", default=None, help="Optional W&B group override.")
    parser.add_argument(
        "--log-shared-conservation-metric",
        action="store_true",
        help="Log eval/conservation_law_nats in addition to the detailed eval payload.",
    )
    return parser


def _resolve_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def discover_config_path(checkpoint_path: str | Path) -> Path:
    checkpoint = _resolve_path(checkpoint_path)
    for candidate in [checkpoint.parent, *checkpoint.parents]:
        config_path = candidate / "config.yaml"
        if config_path.is_file():
            return config_path
    raise FileNotFoundError(f"Could not find config.yaml by walking upward from {checkpoint}.")


def _normalize_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            normalized[key[len("module."):]] = value
        else:
            normalized[key] = value
    return normalized


def _load_checkpoint_payload(checkpoint_path: str | Path) -> dict[str, Any]:
    payload = torch.load(_resolve_path(checkpoint_path), map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint at {checkpoint_path} must deserialize to a dict payload.")
    return payload


def _build_causal_model(cfg: Any, *, vocab_size: int, context_length: int) -> CausalTransformerLM:
    return CausalTransformerLM(
        vocab_size=vocab_size,
        hidden_size=int(cfg.model.hidden_size),
        n_layers=int(cfg.model.n_layers),
        n_heads=int(cfg.model.n_heads),
        ffn_hidden_size=int(cfg.model.ffn_hidden_size),
        max_seq_len=context_length,
        dropout=float(cfg.model.dropout),
        use_timestep_conditioning=False,
    )


def _build_masked_model(cfg: Any, *, vocab_size: int, context_length: int) -> MaskedTransformerLM:
    return MaskedTransformerLM(
        vocab_size=vocab_size,
        hidden_size=int(cfg.model.hidden_size),
        n_layers=int(cfg.model.n_layers),
        n_heads=int(cfg.model.n_heads),
        ffn_hidden_size=int(cfg.model.ffn_hidden_size),
        max_seq_len=context_length,
        dropout=float(cfg.model.dropout),
        use_timestep_conditioning=bool(cfg.model.get("use_timestep_conditioning", False)),
    )


def _build_immse_model(cfg: Any, *, vocab_size: int, context_length: int) -> ContinuousDiffusionTransformerLM:
    return ContinuousDiffusionTransformerLM(
        vocab_size=vocab_size,
        hidden_size=int(cfg.model.hidden_size),
        n_layers=int(cfg.model.n_layers),
        n_heads=int(cfg.model.n_heads),
        ffn_hidden_size=int(cfg.model.ffn_hidden_size),
        max_seq_len=context_length,
        dropout=float(cfg.model.dropout),
    )


def _save_optional_json(path: str | Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    resolved = _resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    configure_environment()
    args = build_arg_parser().parse_args(argv)
    checkpoint_path = _resolve_path(args.checkpoint)
    config_path = _resolve_path(args.config) if args.config is not None else discover_config_path(checkpoint_path)
    output_dir = config_path.parent
    cfg = OmegaConf.load(config_path)
    repo_root = find_repo_root(APP_ROOT)
    benchmark = load_lm_benchmark(cfg.benchmark, project_root=repo_root)
    payload = _load_checkpoint_payload(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    objective = str(cfg.objective)
    split = str(args.split)
    batch_size = int(args.batch_size or cfg.eval.batch_size)
    max_batches = None if args.max_batches is None else int(args.max_batches)
    run_id = str(getattr(cfg.logging, "run_id", None) or output_dir.name)
    progress_label = f"{run_id}/{split}"
    if args.wandb_group is not None:
        cfg.logging.group = str(args.wandb_group)
    cfg.logging.tags = list(dict.fromkeys([*list(cfg.logging.tags), "eval", split]))
    logger = create_rank_aware_logger(
        cfg=cfg,
        run_id=f"{run_id}-eval-{split}",
        output_dir=str(output_dir / cfg.paths.wandb_dir_name),
        enabled=bool(cfg.logging.enabled),
    )

    if str(cfg.kind) == "lm_causal_ce":
        model = _build_causal_model(cfg, vocab_size=benchmark.vocab_size, context_length=benchmark.context_length)
        model.load_state_dict(_normalize_state_dict_keys(payload["model_state_dict"]))
        model = model.to(device)
        model_params = count_parameters(model)
        split_metrics = evaluate_causal_nats(
            model,
            benchmark,
            split=split,
            batch_size=batch_size,
            device=device,
            max_batches=max_batches,
        )
        split_metrics["conservation_law_nats"] = conservation_law_nats(objective=objective, metrics=split_metrics)
        metrics = {
            "checkpoint": str(checkpoint_path),
            "config": str(config_path),
            "run_id": run_id,
            "experiment_kind": str(cfg.kind),
            "objective": objective,
            "benchmark": benchmark.benchmark_metadata(),
            "model": {
                "name": str(cfg.model.name),
                "parameter_count": int(model_params),
                "hidden_size": int(cfg.model.hidden_size),
                "n_layers": int(cfg.model.n_layers),
                "n_heads": int(cfg.model.n_heads),
                "ffn_hidden_size": int(cfg.model.ffn_hidden_size),
                "dropout": float(cfg.model.dropout),
            },
            "train": {
                "n_steps": int(cfg.train.n_steps),
                "batch_size": int(cfg.train.batch_size),
                "grad_accum_steps": int(cfg.train.grad_accum_steps),
                "lr": float(cfg.train.lr),
                "weight_decay": float(cfg.train.weight_decay),
                "max_grad_norm": float(cfg.train.max_grad_norm),
                "tokens_seen": int(payload.get("tokens_seen", -1)),
            },
            "nats_per_token": {split: float(split_metrics["nats_per_token"])},
            "eval_details": {split: split_metrics},
            "ranking_metric_nats_per_token": float(split_metrics["nats_per_token"]),
            "primary_split": split,
            "step": int(payload.get("step", -1)),
            "device": str(device),
        }
        save_json(output_dir / cfg.paths.metrics_filename, metrics)
    elif str(cfg.kind) == "lm_exit":
        model = _build_masked_model(cfg, vocab_size=benchmark.vocab_size, context_length=benchmark.context_length)
        model.load_state_dict(_normalize_state_dict_keys(payload["model_state_dict"]))
        model = model.to(device)
        model_params = count_parameters(model)
        objective_metrics = evaluate_exit_objective_metrics(
            model,
            benchmark,
            split=split,
            batch_size=batch_size,
            device=device,
            min_rate=float(cfg.masking.train_mask_min),
            max_rate=float(cfg.masking.train_mask_max),
            max_batches=max_batches,
            steps_per_batch=int(getattr(cfg.eval, "steps_per_batch", 1)),
        )
        grid, curve_dict, curve_metrics, curve_rows = evaluate_exit_curve(
            model,
            benchmark,
            split=split,
            batch_size=batch_size,
            device=device,
            n_s_points=int(cfg.eval.n_s_points),
            max_batches=max_batches,
            steps_per_batch=int(getattr(cfg.eval, "steps_per_batch", 1)),
            progress_label=f"{progress_label}/exit_curve",
        )
        split_metrics = {**objective_metrics, **curve_metrics}
        split_metrics["mc_conservation_law_nats"] = float(objective_metrics["mc_conservation_law_nats"])
        split_metrics["integral_conservation_law_nats"] = float(split_metrics["curve_integral_nats"])
        split_metrics["conservation_law_nats"] = conservation_law_nats(objective=objective, metrics=split_metrics)
        metrics = {
            "checkpoint": str(checkpoint_path),
            "config": str(config_path),
            "run_id": run_id,
            "experiment_kind": str(cfg.kind),
            "objective": objective,
            "benchmark": benchmark.benchmark_metadata(),
            "model": {
                "name": str(cfg.model.name),
                "parameter_count": int(model_params),
                "hidden_size": int(cfg.model.hidden_size),
                "n_layers": int(cfg.model.n_layers),
                "n_heads": int(cfg.model.n_heads),
                "ffn_hidden_size": int(cfg.model.ffn_hidden_size),
                "dropout": float(cfg.model.dropout),
            },
            "train": {
                "n_steps": int(cfg.train.n_steps),
                "batch_size": int(cfg.train.batch_size),
                "grad_accum_steps": int(cfg.train.grad_accum_steps),
                "lr": float(cfg.train.lr),
                "weight_decay": float(cfg.train.weight_decay),
                "max_grad_norm": float(cfg.train.max_grad_norm),
                "train_mask_min": float(cfg.masking.train_mask_min),
                "train_mask_max": float(cfg.masking.train_mask_max),
                "tokens_seen": int(payload.get("tokens_seen", -1)),
            },
            "nats_per_token": {split: float(split_metrics["nats_per_token"])},
            "eval_details": {split: split_metrics},
            "ranking_metric_nats_per_token": float(split_metrics["nats_per_token"]),
            "primary_split": split,
            "step": int(payload.get("step", -1)),
            "device": str(device),
        }
        save_csv(output_dir / cfg.paths.curve_filename, curve_rows)
        save_exit_curve_plot(
            output_dir / cfg.paths.curve_plot_filename,
            s_grid=grid,
            exit_curve=curve_dict["exit_gap"],
            title=f"{benchmark.name} EXIT curve ({split})",
            area=float(split_metrics["curve_integral_nats"]),
        )
        save_json(output_dir / cfg.paths.metrics_filename, metrics)
        logger.log_image("eval/exit_curve", output_dir / cfg.paths.curve_plot_filename, caption=f"split={split}")
    elif str(cfg.kind) == "lm_immse":
        model = _build_immse_model(cfg, vocab_size=benchmark.vocab_size, context_length=benchmark.context_length)
        model.load_state_dict(_normalize_state_dict_keys(payload["model_state_dict"]))
        model = model.to(device)
        model_params = count_parameters(model)
        loss_name = str(getattr(getattr(cfg, "loss", None), "name", "nll")).lower()
        weight_loss_by_snr = bool(getattr(getattr(cfg, "loss", None), "weight_loss_by_snr", True))
        objective_metrics = evaluate_immse_objective_metrics(
            model,
            benchmark,
            split=split,
            batch_size=batch_size,
            device=device,
            snr_min=float(cfg.train.snr_min),
            snr_max=float(cfg.train.snr_max),
            loss_name=loss_name,
            weight_loss_by_snr=weight_loss_by_snr,
            max_batches=max_batches,
            steps_per_batch=int(getattr(cfg.eval, "steps_per_batch", 1)),
        )
        grid, curve, curve_metrics, curve_rows = evaluate_immse_curve(
            model,
            benchmark,
            split=split,
            batch_size=batch_size,
            device=device,
            snr_min=float(cfg.eval.snr_min),
            snr_max=float(cfg.eval.snr_max),
            n_snr_points=int(cfg.eval.n_snr_points),
            max_batches=max_batches,
            steps_per_batch=int(getattr(cfg.eval, "steps_per_batch", 1)),
            progress_label=f"{progress_label}/immse_curve",
        )
        split_metrics = {**objective_metrics, **curve_metrics}
        split_metrics["mc_conservation_law_nats"] = float(objective_metrics["mc_conservation_law_nats"])
        split_metrics["integral_conservation_law_nats"] = float(split_metrics["curve_integral_nats"])
        split_metrics["conservation_law_nats"] = conservation_law_nats(objective=objective, metrics=split_metrics)
        metrics = {
            "checkpoint": str(checkpoint_path),
            "config": str(config_path),
            "run_id": run_id,
            "experiment_kind": str(cfg.kind),
            "objective": objective,
            "loss_name": loss_name,
            "benchmark": benchmark.benchmark_metadata(),
            "model": {
                "name": str(cfg.model.name),
                "parameter_count": int(model_params),
                "hidden_size": int(cfg.model.hidden_size),
                "n_layers": int(cfg.model.n_layers),
                "n_heads": int(cfg.model.n_heads),
                "ffn_hidden_size": int(cfg.model.ffn_hidden_size),
                "dropout": float(cfg.model.dropout),
            },
            "train": {
                "n_steps": int(cfg.train.n_steps),
                "batch_size": int(cfg.train.batch_size),
                "grad_accum_steps": int(cfg.train.grad_accum_steps),
                "lr": float(cfg.train.lr),
                "weight_decay": float(cfg.train.weight_decay),
                "max_grad_norm": float(cfg.train.max_grad_norm),
                "snr_min": float(cfg.train.snr_min),
                "snr_max": float(cfg.train.snr_max),
                "tokens_seen": int(payload.get("tokens_seen", -1)),
            },
            "nats_per_token": {split: float(split_metrics["nats_per_token"])},
            "eval_details": {split: split_metrics},
            "ranking_metric_nats_per_token": float(split_metrics["nats_per_token"]),
            "primary_split": split,
            "step": int(payload.get("step", -1)),
            "device": str(device),
        }
        save_csv(output_dir / cfg.paths.curve_filename, curve_rows)
        save_immse_curve_plot(
            output_dir / cfg.paths.curve_plot_filename,
            snr_grid=grid,
            mmse_curve=curve,
            title=f"{benchmark.name} I-MMSE curve ({split})",
            area=float(split_metrics["curve_integral_nats"]),
        )
        save_json(output_dir / cfg.paths.metrics_filename, metrics)
        logger.log_image("eval/mmse_curve", output_dir / cfg.paths.curve_plot_filename, caption=f"split={split}")
    elif str(cfg.kind) == "lm_qsc":
        model = _build_masked_model(cfg, vocab_size=benchmark.vocab_size, context_length=benchmark.context_length)
        model.load_state_dict(_normalize_state_dict_keys(payload["model_state_dict"]))
        model = model.to(device)
        model_params = count_parameters(model)
        objective_metrics = evaluate_qsc_objective_metrics(
            model,
            benchmark,
            split=split,
            batch_size=batch_size,
            device=device,
            t_min=float(cfg.eval.t_min),
            t_max=float(cfg.eval.t_max),
            max_batches=max_batches,
            steps_per_batch=int(getattr(cfg.eval, "steps_per_batch", 1)),
        )
        grid, curve_dict, curve_metrics, curve_rows = evaluate_qsc_curve(
            model,
            benchmark,
            split=split,
            batch_size=batch_size,
            device=device,
            t_min=float(cfg.eval.t_min),
            t_max=float(cfg.eval.t_max),
            n_t_points=int(cfg.eval.n_t_points),
            max_batches=max_batches,
            steps_per_batch=int(getattr(cfg.eval, "steps_per_batch", 1)),
            progress_label=f"{progress_label}/qsc_curve",
        )
        split_metrics = {**objective_metrics, **curve_metrics}
        t_width = float(cfg.eval.t_max) - float(cfg.eval.t_min)
        split_metrics["mc_conservation_law_nats"] = float(objective_metrics["conservation_law_nats"])
        split_metrics["integral_conservation_law_nats"] = float(split_metrics["curve_integral_nats"])
        split_metrics["qsc_gexit_mc_nats"] = float(objective_metrics["qsc_gexit_nats"])
        split_metrics["qsc_gexit_integral_nats"] = float(split_metrics["curve_integral_nats"]) / t_width
        split_metrics["qsc_gexit_nats"] = float(split_metrics["qsc_gexit_integral_nats"])
        split_metrics["conservation_law_nats"] = conservation_law_nats(objective=objective, metrics=split_metrics)
        metrics = {
            "checkpoint": str(checkpoint_path),
            "config": str(config_path),
            "run_id": run_id,
            "experiment_kind": str(cfg.kind),
            "objective": objective,
            "benchmark": benchmark.benchmark_metadata(),
            "model": {
                "name": str(cfg.model.name),
                "parameter_count": int(model_params),
                "hidden_size": int(cfg.model.hidden_size),
                "n_layers": int(cfg.model.n_layers),
                "n_heads": int(cfg.model.n_heads),
                "ffn_hidden_size": int(cfg.model.ffn_hidden_size),
                "dropout": float(cfg.model.dropout),
            },
            "train": {
                "n_steps": int(cfg.train.n_steps),
                "batch_size": int(cfg.train.batch_size),
                "grad_accum_steps": int(cfg.train.grad_accum_steps),
                "lr": float(cfg.train.lr),
                "weight_decay": float(cfg.train.weight_decay),
                "max_grad_norm": float(cfg.train.max_grad_norm),
                "train_t_min": float(cfg.qsc.train_t_min),
                "train_t_max": float(cfg.qsc.train_t_max),
                "tokens_seen": int(payload.get("tokens_seen", -1)),
            },
            "nats_per_token": {split: float(split_metrics["nats_per_token"])},
            "eval_details": {split: split_metrics},
            "ranking_metric_nats_per_token": float(split_metrics["nats_per_token"]),
            "primary_split": split,
            "step": int(payload.get("step", -1)),
            "device": str(device),
        }
        save_csv(output_dir / cfg.paths.curve_filename, curve_rows)
        save_qsc_curve_plot(
            output_dir / cfg.paths.curve_plot_filename,
            t_grid=grid,
            qsc_curve=curve_dict["qsc_gexit"],
            title=f"{benchmark.name} QSC curve ({split})",
            area=float(split_metrics["curve_integral_nats"]),
        )
        save_json(output_dir / cfg.paths.metrics_filename, metrics)
        logger.log_image("eval/qsc_curve", output_dir / cfg.paths.curve_plot_filename, caption=f"split={split}")
    else:
        raise ValueError(f"Unsupported kind={cfg.kind!r}.")

    final_payload: dict[str, Any] = {}
    if isinstance(metrics.get("eval_details"), dict):
        for split_name, split_values in metrics["eval_details"].items():
            final_payload.update(prefixed_scalar_metrics(str(split_name), split_values))
    if final_payload:
        logger.log(final_payload, step=int(metrics.get("step", -1)))
    if args.log_shared_conservation_metric:
        logger.log({"eval/conservation_law_nats": float(metrics["eval_details"][split]["conservation_law_nats"])}, step=int(metrics.get("step", -1)))
    logger.log({"eval": metrics})
    logger.log_directory_files(output_dir, artifact_name=f"{run_id}-eval-outputs", exclude_dirs=[str(cfg.paths.wandb_dir_name)])
    _save_optional_json(args.output_json, metrics)
    logger.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
