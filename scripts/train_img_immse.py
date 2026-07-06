from __future__ import annotations

import math
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from experiments.common.io import save_config, save_csv, save_json
from experiments.common.repro import configure_environment, set_global_seed
from scripts.img_common import (
    attach_bpd_to_split_metrics,
    compose_img_config,
    conservation_law_nats,
    count_parameters,
    create_rank_aware_logger,
    eval_generate_curves,
    eval_log_conservation_integral,
    find_app_root,
    find_repo_root,
    lm_embedding_norm,
    load_checkpoint_if_requested,
    load_history_prefix,
    load_img_benchmark,
    maybe_apply_lm_lr_schedule,
    maybe_reset_peak_memory,
    resolve_project_path,
    resolve_output_dir,
    resolve_test_split_name,
    resolve_valid_split_name,
    runtime_metrics,
    save_discrete_reconstruction_panel,
    save_training_checkpoint,
)
from src.channels import awgn, sample_log_uniform_snr
from src.img_models import RGB8ContinuousDDPMModel, RGB8DistributionDDPMModel
from src.lm_distributed import all_reduce_mean_scalars, all_reduce_sum_scalars, barrier, cleanup_distributed, setup_distributed, unwrap_model
from src.lm_objectives import compute_immse_loss, evaluate_immse_curve, evaluate_immse_objective_metrics, save_immse_curve_plot


def _validate_cfg(cfg) -> None:
    if str(cfg.kind) != "img_immse":
        raise ValueError(f"Expected kind='img_immse', got {cfg.kind!r}.")
    if float(cfg.train.snr_min) <= 0.0 or float(cfg.train.snr_max) <= 0.0:
        raise ValueError("train.snr_min and train.snr_max must be strictly positive.")
    if str(cfg.model.name).lower() not in {"img_unet_rgb8_continuous", "img_unet_rgb8_distribution"}:
        raise ValueError("CIFAR image I-MMSE release configs require an img_unet_rgb8_* model.")
    if str(cfg.benchmark.name).lower() != "cifar10_rgb8_tokens":
        raise ValueError(f"{cfg.model.name} requires benchmark.name='cifar10_rgb8_tokens'.")


def _build_model(cfg, *, vocab_size: int, context_length: int):
    if str(cfg.model.name).lower() == "img_unet_rgb8_continuous":
        return RGB8ContinuousDDPMModel(
            vocab_size=vocab_size,
            image_size=int(cfg.benchmark.image_size),
            base_channels=int(cfg.model.base_channels),
            channel_multipliers=tuple(int(value) for value in cfg.model.channel_multipliers),
            dropout=float(cfg.model.dropout),
            time_embed_input_dim=int(cfg.model.time_embed_input_dim),
            time_embed_dim=int(cfg.model.time_embed_dim),
            input_skip=bool(cfg.model.get("input_skip", False)),
        )
    if str(cfg.model.name).lower() == "img_unet_rgb8_distribution":
        return RGB8DistributionDDPMModel(
            vocab_size=vocab_size,
            image_size=int(cfg.benchmark.image_size),
            base_channels=int(cfg.model.base_channels),
            channel_multipliers=tuple(int(value) for value in cfg.model.channel_multipliers),
            dropout=float(cfg.model.dropout),
            time_embed_input_dim=int(cfg.model.time_embed_input_dim),
            time_embed_dim=int(cfg.model.time_embed_dim),
        )
    raise ValueError(f"Unsupported CIFAR image I-MMSE model.name={cfg.model.name!r}.")


def _model_metadata(cfg, *, model_params: int) -> dict[str, object]:
    model_info: dict[str, object] = {
        "name": str(cfg.model.name),
        "parameter_count": int(model_params),
        "dropout": float(cfg.model.dropout),
        "base_channels": int(cfg.model.base_channels),
        "channel_multipliers": [int(value) for value in cfg.model.channel_multipliers],
        "time_embed_input_dim": int(cfg.model.time_embed_input_dim),
        "time_embed_dim": int(cfg.model.time_embed_dim),
    }
    return model_info


@torch.no_grad()
def _init_ema_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone().cpu() for name, value in unwrap_model(model).state_dict().items()}


@torch.no_grad()
def _load_ema_state_if_present(
    *,
    checkpoint_path: str | Path | None,
    project_root: Path,
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    resolved = None if checkpoint_path in (None, "") else Path(str(checkpoint_path)).expanduser()
    if resolved is not None and not resolved.is_absolute():
        resolved = (project_root / resolved).resolve()
    if resolved is None or not resolved.is_file():
        return _init_ema_state(model)
    payload = torch.load(resolved, map_location="cpu")
    if isinstance(payload, dict) and isinstance(payload.get("ema_state_dict"), dict):
        return {str(name): torch.as_tensor(value).detach().clone().cpu() for name, value in payload["ema_state_dict"].items()}
    return _init_ema_state(model)


@torch.no_grad()
def _update_ema_state(model: torch.nn.Module, ema_state: dict[str, torch.Tensor], decay: float) -> None:
    current_state = unwrap_model(model).state_dict()
    for name, value in current_state.items():
        value_cpu = value.detach().cpu()
        if name not in ema_state:
            ema_state[name] = value_cpu.clone()
            continue
        if torch.is_floating_point(value_cpu):
            ema_state[name].mul_(float(decay)).add_(value_cpu, alpha=1.0 - float(decay))
        else:
            ema_state[name].copy_(value_cpu)


def _extract_state_dict(payload, *, source_key: str | None) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping) and source_key:
        candidate = payload.get(source_key)
        if not isinstance(candidate, Mapping):
            raise KeyError(f"Checkpoint does not contain state dict key {source_key!r}.")
        return candidate
    if isinstance(payload, Mapping):
        for key in ("model_state_dict", "state_dict"):
            candidate = payload.get(key)
            if isinstance(candidate, Mapping):
                return candidate
        if all(isinstance(name, str) for name in payload):
            return payload
    raise TypeError("Could not extract a model state dict from checkpoint payload.")


def _load_initial_weights_if_requested(
    *,
    model: torch.nn.Module,
    checkpoint_path: str | Path | None,
    project_root: Path,
    source_key: str | None,
    strict: bool,
) -> tuple[Path | None, dict[str, int]]:
    resolved = resolve_project_path(project_root, checkpoint_path)
    empty_stats = {"loaded": 0, "skipped_missing": 0, "skipped_shape": 0}
    if resolved is None:
        return None, empty_stats
    if not resolved.is_file():
        raise FileNotFoundError(f"Initial weights checkpoint not found: {resolved}")
    payload = torch.load(resolved, map_location="cpu")
    state_dict = _extract_state_dict(payload, source_key=source_key)
    base_model = unwrap_model(model)
    if strict:
        base_model.load_state_dict(state_dict, strict=True)
        return resolved, {"loaded": len(state_dict), "skipped_missing": 0, "skipped_shape": 0}

    current_state = base_model.state_dict()
    filtered_state = {}
    skipped_missing = 0
    skipped_shape = 0
    for key, value in state_dict.items():
        if key not in current_state:
            skipped_missing += 1
            continue
        if tuple(current_state[key].shape) != tuple(value.shape):
            skipped_shape += 1
            continue
        filtered_state[key] = value
    base_model.load_state_dict(filtered_state, strict=False)
    return resolved, {"loaded": len(filtered_state), "skipped_missing": skipped_missing, "skipped_shape": skipped_shape}


def _normalize_state_dict_keys(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key[len("module.") :] if key.startswith("module.") else key: value for key, value in state_dict.items()}


def _load_initial_weights_from_exit_checkpoint(
    *,
    model: torch.nn.Module,
    exit_checkpoint_path: str | Path | None,
    project_root: Path,
    source_key: str | None,
) -> tuple[Path | None, dict[str, int]]:
    resolved = resolve_project_path(project_root, exit_checkpoint_path)
    empty_stats = {"loaded": 0, "skipped_missing": 0, "skipped_shape": 0, "mapped": 0}
    if resolved is None:
        return None, empty_stats
    if not resolved.is_file():
        raise FileNotFoundError(f"Exit init checkpoint not found: {resolved}")
    payload = torch.load(resolved, map_location="cpu")
    exit_state = _normalize_state_dict_keys(_extract_state_dict(payload, source_key=source_key))
    base_model = unwrap_model(model)
    current_state = base_model.state_dict()

    mapped_state: dict[str, torch.Tensor] = {}
    mapped = 0
    # DDPM-style EXIT checkpoints (RGB8ImageTokenUNet) share the UNet backbone module.
    for key, value in exit_state.items():
        if not key.startswith("backbone."):
            continue
        if key not in current_state:
            continue
        if tuple(current_state[key].shape) != tuple(value.shape):
            continue
        mapped_state[key] = value
        mapped += 1
    if "token_embedding.weight" in exit_state and "x_values.weight" in current_state:
        token_weight = exit_state["token_embedding.weight"]
        x_weight = current_state["x_values.weight"]
        if token_weight.ndim == 2 and x_weight.ndim == 2 and token_weight.shape[0] >= x_weight.shape[0] and token_weight.shape[1] == x_weight.shape[1]:
            mapped_state["x_values.weight"] = token_weight[: x_weight.shape[0]].detach().clone()
            mapped += 1

    rename_map = {
        "backbone_base.backbone": "backbone_base.backbone",
        "backbone_base.row_embedding": "backbone_base.row_embedding",
        "backbone_base.col_embedding": "backbone_base.col_embedding",
        "backbone_base.channel_embedding": "backbone_base.channel_embedding",
        "t_embedder": "t_embedder",
        "logits_bias": "logits_bias",
    }
    for prefix_src, prefix_dst in rename_map.items():
        for key, value in exit_state.items():
            if key == prefix_src or key.startswith(prefix_src + "."):
                dst_key = prefix_dst + key[len(prefix_src) :]
                if dst_key not in current_state:
                    continue
                if tuple(current_state[dst_key].shape) != tuple(value.shape):
                    continue
                mapped_state[dst_key] = value
                mapped += 1

    skipped_missing = 0
    skipped_shape = 0
    loaded = 0
    for key, value in mapped_state.items():
        if key not in current_state:
            skipped_missing += 1
            continue
        if tuple(current_state[key].shape) != tuple(value.shape):
            skipped_shape += 1
            continue
        loaded += 1
    base_model.load_state_dict(mapped_state, strict=False)
    return resolved, {"loaded": loaded, "skipped_missing": skipped_missing, "skipped_shape": skipped_shape, "mapped": mapped}


def main(argv: list[str] | None = None) -> int:
    configure_environment()
    argv = sys.argv[1:] if argv is None else argv
    app_root = find_app_root(Path(__file__).resolve())
    repo_root = find_repo_root(app_root)
    config_name, overrides, cfg = compose_img_config(project_root=app_root, argv=argv)
    _validate_cfg(cfg)

    dist_ctx, device = setup_distributed(str(cfg.device))
    set_global_seed(int(cfg.seed) + int(dist_ctx.rank), deterministic=bool(cfg.deterministic))
    benchmark = load_img_benchmark(cfg.benchmark, project_root=repo_root)
    train_token_count = int(benchmark.token_counts()["train"])
    run_id, output_dir = resolve_output_dir(project_root=app_root, cfg=cfg, config_name=config_name, overrides=overrides)
    if dist_ctx.is_main_process:
        save_config(output_dir / cfg.paths.config_snapshot_name, cfg)

    logger = create_rank_aware_logger(cfg=cfg, run_id=run_id, output_dir=str(output_dir / cfg.paths.wandb_dir_name), enabled=dist_ctx.is_main_process)
    model = _build_model(cfg, vocab_size=benchmark.vocab_size, context_length=benchmark.context_length).to(device)
    if cfg.train.get("init_weights_path") is not None and cfg.train.get("init_exit_checkpoint_path") is not None:
        raise ValueError("Set only one of train.init_weights_path or train.init_exit_checkpoint_path for IMMSE runs.")
    initialized_from = None
    init_stats: dict[str, int] = {"loaded": 0, "skipped_missing": 0, "skipped_shape": 0}
    if cfg.train.get("init_exit_checkpoint_path") is not None:
        initialized_from, init_stats = _load_initial_weights_from_exit_checkpoint(
            model=model,
            exit_checkpoint_path=cfg.train.get("init_exit_checkpoint_path"),
            project_root=app_root,
            source_key=cfg.train.get("init_exit_checkpoint_source"),
        )
    else:
        initialized_from, init_stats = _load_initial_weights_if_requested(
            model=model,
            checkpoint_path=cfg.train.get("init_weights_path"),
            project_root=app_root,
            source_key=cfg.train.get("init_weights_source"),
            strict=bool(cfg.train.get("init_weights_strict", True)),
        )
    if initialized_from is not None and dist_ctx.is_main_process:
        print(
            f"[setup] initialized model weights from {initialized_from} "
            f"(loaded={init_stats['loaded']} skipped_missing={init_stats['skipped_missing']} skipped_shape={init_stats['skipped_shape']})",
            flush=True,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay), betas=(float(cfg.train.beta1), float(cfg.train.beta2)))
    start_step, tokens_seen = load_checkpoint_if_requested(model=model, optimizer=optimizer, checkpoint_path=cfg.train.resume_from, project_root=app_root)
    ema_decay = cfg.train.get("ema_decay", None)
    ema_state = None
    if ema_decay not in (None, ""):
        ema_decay = float(ema_decay)
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError(f"train.ema_decay must be in [0, 1), got {ema_decay}.")
        ema_state = _load_ema_state_if_present(checkpoint_path=cfg.train.resume_from, project_root=app_root, model=model)
    if dist_ctx.enabled:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None, output_device=device.index if device.type == "cuda" else None)
    model_params = count_parameters(model)
    logger.log({"model/parameter_count": int(model_params)})

    history_rows: list[dict[str, float | int | str]] = load_history_prefix(output_dir / cfg.paths.history_filename, start_step=start_step)
    train_generator = torch.Generator(device="cpu")
    train_generator.manual_seed(int(cfg.seed) + 707 + int(dist_ctx.rank) * 1_000)
    maybe_reset_peak_memory(device)
    start_time = time.perf_counter()
    log_snr_width = math.log(float(cfg.train.snr_max)) - math.log(float(cfg.train.snr_min))
    high_snr_quantile = 0.99
    high_snr_threshold = math.exp(math.log(float(cfg.train.snr_min)) + high_snr_quantile * log_snr_width)
    valid_split = resolve_valid_split_name(benchmark)
    test_split = resolve_test_split_name(benchmark)
    valid_conservation_every = int(cfg.eval.eval_every)
    loss_name = str(getattr(getattr(cfg, "loss", None), "name", "nll")).lower()
    weight_loss_by_snr = bool(getattr(getattr(cfg, "loss", None), "weight_loss_by_snr", True))

    for step in range(start_step, int(cfg.train.n_steps) + 1):
        model.train()
        maybe_apply_lm_lr_schedule(cfg, step=step, optimizer=optimizer)
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        conservation_total = 0.0
        mse_total = 0.0
        posterior_mismatched_total = 0.0
        high_snr_mse_sum_total = 0.0
        high_snr_token_count_total = 0.0
        local_tokens_this_step = 0

        for _ in range(int(cfg.train.grad_accum_steps)):
            tokens = benchmark.sample_train_batch(batch_size=int(cfg.train.batch_size), device=device, include_next_token=False, generator=train_generator)
            base_model = unwrap_model(model)
            if bool(getattr(base_model, "expects_per_image_snr", False)):
                snr = sample_log_uniform_snr((int(cfg.train.batch_size), 1, 1), snr_min=float(cfg.train.snr_min), snr_max=float(cfg.train.snr_max), device=device, generator=train_generator)
                snr = snr.expand(int(cfg.train.batch_size), benchmark.context_length, 1)
            else:
                snr = sample_log_uniform_snr((int(cfg.train.batch_size), benchmark.context_length, 1), snr_min=float(cfg.train.snr_min), snr_max=float(cfg.train.snr_max), device=device, generator=train_generator)
            stats = compute_immse_loss(model, tokens=tokens, snr=snr, log_snr_width=log_snr_width, label_smoothing=float(cfg.train.label_smoothing), loss_name=loss_name, weight_loss_by_snr=weight_loss_by_snr, high_snr_threshold=high_snr_threshold)
            (stats["loss"] / int(cfg.train.grad_accum_steps)).backward()
            loss_total += float(stats["loss"].item())
            conservation_total += float(stats["conservation_law"].item())
            mse_total += float(stats["mse"].item())
            posterior_mismatched_total += float(stats["posterior_mismatched"].item())
            high_snr_mse_sum_total += float(stats["high_snr_mse_sum"].item())
            high_snr_token_count_total += float(stats["high_snr_token_count"].item())
            local_tokens_this_step += int(tokens.numel())

        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.train.max_grad_norm)).item())
        optimizer.step()
        if ema_state is not None:
            _update_ema_state(model, ema_state, decay=float(ema_decay))
        reduced_loss, reduced_conservation, reduced_mse, reduced_posterior_mismatched, reduced_grad_norm, reduced_embed_norm = all_reduce_mean_scalars(
            dist_ctx,
            [
                loss_total / int(cfg.train.grad_accum_steps),
                conservation_total / int(cfg.train.grad_accum_steps),
                mse_total / int(cfg.train.grad_accum_steps),
                posterior_mismatched_total / int(cfg.train.grad_accum_steps),
                grad_norm,
                lm_embedding_norm(model),
            ],
            device=device,
        )
        global_high_snr_mse_sum, global_high_snr_token_count = all_reduce_sum_scalars(
            dist_ctx,
            [high_snr_mse_sum_total, high_snr_token_count_total],
            device=device,
        )
        high_snr_mse = float(global_high_snr_mse_sum / global_high_snr_token_count) if float(global_high_snr_token_count) > 0.0 else float("nan")
        global_tokens_this_step = int(all_reduce_sum_scalars(dist_ctx, [float(local_tokens_this_step)], device=device)[0])
        tokens_seen += int(global_tokens_this_step)
        row = {
            "step": int(step),
            "train_loss_nats": float(reduced_loss),
            "train_conservation_law_nats": float(reduced_conservation),
            "train_mse": float(reduced_mse),
            "train_high_snr_mse": high_snr_mse,
            "train_high_snr_token_count": int(global_high_snr_token_count),
            "train_high_snr_threshold": float(high_snr_threshold),
            "train_posterior_mismatched": float(reduced_posterior_mismatched),
            "grad_norm": float(reduced_grad_norm),
            "embed_norm": float(reduced_embed_norm),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "tokens_seen": int(tokens_seen),
            "epoch_progress": float(tokens_seen / max(train_token_count, 1)),
            "val_objective_loss_nats": None,
            "val_mse": None,
            "val_nll": None,
            "val_nll_ext": None,
            "val_conservation_law_nats": None,
            "val_conservation_law_mc_nats": None,
            "val_conservation_law_integral_nats": None,
            "valid_conservation_law_nats": None,
            "val_evaluated_tokens": None,
            "val_evaluated_batches": None,
        }
        history_rows.append(row)
        if step == start_step or step % int(cfg.logging.log_every) == 0 or step == int(cfg.train.n_steps):
            logger.log(
                {
                    "train/loss": row["train_loss_nats"],
                    "train/conservation_law": row["train_conservation_law_nats"],
                    "train/mse": row["train_mse"],
                    "train/high_snr_mse": row["train_high_snr_mse"],
                    "train/high_snr_token_count": row["train_high_snr_token_count"],
                    "train/high_snr_threshold": row["train_high_snr_threshold"],
                    "train/grad_norm": row["grad_norm"],
                    "train/embed_norm": row["embed_norm"],
                    "train/embedding_norm": row["embed_norm"],
                    "train/lr": row["lr"],
                    "train/tokens_seen": row["tokens_seen"],
                    "train/epoch_progress": row["epoch_progress"],
                },
                step=step,
            )
        if dist_ctx.is_main_process and (step % int(cfg.train.save_every) == 0 or step == int(cfg.train.n_steps)):
            save_training_checkpoint(
                output_dir=output_dir,
                checkpoint_dir_name=str(cfg.paths.checkpoint_dir_name),
                filename_template=str(cfg.paths.checkpoint_filename_template),
                step=step,
                model=model,
                optimizer=optimizer,
                tokens_seen=tokens_seen,
                extra_state=None if ema_state is None else {"ema_decay": float(ema_decay), "ema_state_dict": ema_state},
            )

        if eval_log_conservation_integral(cfg) and step % valid_conservation_every == 0 and step < int(cfg.train.n_steps):
            val_metrics = evaluate_immse_objective_metrics(
                model,
                benchmark,
                split=valid_split,
                batch_size=int(cfg.eval.batch_size),
                device=device,
                snr_min=float(cfg.train.snr_min),
                snr_max=float(cfg.train.snr_max),
                loss_name=loss_name,
                weight_loss_by_snr=weight_loss_by_snr,
                max_batches=None if cfg.eval.max_batches is None else int(cfg.eval.max_batches),
                steps_per_batch=int(getattr(cfg.eval, "steps_per_batch", 1)),
                seed=int(cfg.seed) + 16_500 + step,
                dist_ctx=dist_ctx,
            )
            _valid_snr_grid, _valid_mmse_curve, valid_curve_metrics, _valid_curve_rows = evaluate_immse_curve(
                model,
                benchmark,
                split=valid_split,
                batch_size=int(cfg.eval.batch_size),
                device=device,
                snr_min=float(cfg.eval.snr_min),
                snr_max=float(cfg.eval.snr_max),
                n_snr_points=int(cfg.eval.n_snr_points),
                max_batches=None if cfg.eval.max_batches is None else int(cfg.eval.max_batches),
                steps_per_batch=int(getattr(cfg.eval, "steps_per_batch", 1)),
                seed=int(cfg.seed) + 17_500 + step,
                dist_ctx=dist_ctx,
            )
            row["valid_conservation_law_nats"] = float(valid_curve_metrics["curve_integral_nats"])
            row["val_conservation_law_nats"] = row["valid_conservation_law_nats"]
            row["val_conservation_law_mc_nats"] = float(val_metrics["mc_conservation_law_nats"])
            row["val_conservation_law_integral_nats"] = row["valid_conservation_law_nats"]
            logger.log(
                {
                    "valid/loss": float(val_metrics["loss_nats"]),
                    "valid/conservation_law": row["valid_conservation_law_nats"],
                    "valid/conservation_law_mc": row["val_conservation_law_mc_nats"],
                    "valid/conservation_law_integral": row["val_conservation_law_integral_nats"],
                    "valid/evaluated_tokens": int(val_metrics["evaluated_tokens"]),
                    "valid/evaluated_sequences": int(val_metrics["evaluated_sequences"]),
                    "valid/mse": float(val_metrics["mse"]),
                },
                step=step,
            )

    curve_split = test_split
    split_metrics: dict[str, dict[str, float]] = {}
    primary_curve_rows: list[dict[str, float]] = []
    primary_curve = None
    primary_grid = None
    dense_integral = None
    for index, split in enumerate([valid_split] if valid_split == test_split else [valid_split, test_split]):
        split_name = str(split)
        objective_metrics = evaluate_immse_objective_metrics(
            model, benchmark, split=split_name, batch_size=int(cfg.eval.batch_size), device=device,
            snr_min=float(cfg.train.snr_min), snr_max=float(cfg.train.snr_max), loss_name=loss_name,
            weight_loss_by_snr=weight_loss_by_snr, max_batches=None if cfg.eval.max_batches is None else int(cfg.eval.max_batches),
            steps_per_batch=int(getattr(cfg.eval, "steps_per_batch", 1)),
            seed=int(cfg.seed) + 15_000 + index * 1000, dist_ctx=dist_ctx,
        )
        snr_grid, mmse_curve, metrics, curve_rows = evaluate_immse_curve(
            model, benchmark, split=split_name, batch_size=int(cfg.eval.batch_size), device=device,
            snr_min=float(cfg.eval.snr_min), snr_max=float(cfg.eval.snr_max), n_snr_points=int(cfg.eval.n_snr_points),
            max_batches=None if cfg.eval.max_batches is None else int(cfg.eval.max_batches), steps_per_batch=int(getattr(cfg.eval, "steps_per_batch", 1)), seed=int(cfg.seed) + 20_000 + index * 1000, dist_ctx=dist_ctx,
        )
        split_metrics[split_name] = {**objective_metrics, **metrics}
        split_metrics[split_name]["mc_conservation_law_nats"] = float(objective_metrics["mc_conservation_law_nats"])
        split_metrics[split_name]["integral_conservation_law_nats"] = float(split_metrics[split_name]["curve_integral_nats"])
        split_metrics[split_name]["conservation_law_nats"] = conservation_law_nats(objective=str(cfg.objective), metrics=split_metrics[split_name])
        if split_name == curve_split:
            primary_curve_rows = curve_rows
            primary_curve = mmse_curve
            primary_grid = snr_grid
            _, _, dense_metrics, _ = evaluate_immse_curve(
                model, benchmark, split=split_name, batch_size=int(cfg.eval.batch_size), device=device,
                snr_min=float(cfg.eval.snr_min), snr_max=float(cfg.eval.snr_max), n_snr_points=int(cfg.eval.dense_n_snr_points),
                max_batches=None if cfg.eval.max_batches is None else int(cfg.eval.max_batches), steps_per_batch=int(getattr(cfg.eval, "steps_per_batch", 1)), seed=int(cfg.seed) + 30_000, dist_ctx=dist_ctx,
            )
            dense_integral = float(dense_metrics["curve_integral_nats"])

    assert primary_curve is not None and primary_grid is not None and dense_integral is not None
    split_metrics[curve_split]["dense_curve_integral_nats"] = float(dense_integral)
    split_metrics[curve_split]["dense_grid_delta_nats"] = float(abs(dense_integral - float(split_metrics[curve_split]["curve_integral_nats"])))
    attach_bpd_to_split_metrics(benchmark=benchmark, objective=str(cfg.objective), split_metrics=split_metrics)
    metrics = {
        "run_id": run_id,
        "experiment_kind": str(cfg.kind),
        "objective": str(cfg.objective),
        "loss_name": str(loss_name),
        "benchmark": benchmark.benchmark_metadata(),
        "model": _model_metadata(cfg, model_params=model_params),
        "train": {
            "n_steps": int(cfg.train.n_steps),
            "batch_size": int(cfg.train.batch_size),
            "grad_accum_steps": int(cfg.train.grad_accum_steps),
            "lr": float(cfg.train.lr),
            "weight_decay": float(cfg.train.weight_decay),
            "snr_min": float(cfg.train.snr_min),
            "snr_max": float(cfg.train.snr_max),
            "tokens_seen": int(tokens_seen),
        },
        "nats_per_token": {split: float(values["nats_per_token"]) for split, values in split_metrics.items()},
        "bpd": {split: float(values["bpd"]) for split, values in split_metrics.items() if "bpd" in values},
        "eval_details": split_metrics,
        "ranking_metric_nats_per_token": float(split_metrics[curve_split]["nats_per_token"]),
        "ranking_metric_bpd": float(split_metrics[curve_split]["bpd"]),
        "primary_split": curve_split,
        **runtime_metrics(start_time=start_time, device=device),
    }
    if dist_ctx.is_main_process:
        save_csv(output_dir / cfg.paths.curve_filename, primary_curve_rows)
        save_immse_curve_plot(output_dir / cfg.paths.curve_plot_filename, snr_grid=primary_grid, mmse_curve=primary_curve, title=f"{benchmark.name} gEXIT curve ({curve_split}, Gaussian noise)", area=float(split_metrics[curve_split]["curve_integral_nats"]))
        sample_batch = next(benchmark.iter_eval_batches(split=curve_split, batch_size=4, device=device, include_next_token=False, max_batches=1))
        clean_images = benchmark.split_images(curve_split)
        clean_images = None if clean_images is None else clean_images[: int(sample_batch.shape[0])]
        sample_snr = torch.full((int(sample_batch.shape[0]), benchmark.context_length, 1), 0.1, dtype=torch.float32, device=device)
        clean = unwrap_model(model).x_values(sample_batch.long())
        noisy = awgn(clean, sample_snr)
        _, _, logits = model(noisy, sample_snr)
        predicted_tokens = torch.argmax(logits, dim=-1)
        distances = torch.cdist(noisy.reshape(-1, noisy.shape[-1]).cpu(), unwrap_model(model).x_values.weight.detach().cpu())
        observed_tokens = torch.argmin(distances, dim=1).view(sample_batch.shape[0], sample_batch.shape[1])
        save_discrete_reconstruction_panel(benchmark, clean_tokens=sample_batch.cpu(), observed_tokens=observed_tokens.cpu(), predicted_tokens=predicted_tokens.cpu(), output_path=output_dir / cfg.paths.reconstruction_plot_filename, clean_images=clean_images)
        save_csv(output_dir / cfg.paths.history_filename, history_rows)
        save_json(output_dir / cfg.paths.metrics_filename, metrics)
        torch.save(unwrap_model(model).state_dict(), output_dir / cfg.paths.model_filename)
        if ema_state is not None:
            torch.save(ema_state, output_dir / "model_ema.pt")
        logger.log(
            {
                "test/loss": float(split_metrics[curve_split]["loss_nats"]),
                "test/conservation_law": float(split_metrics[curve_split]["conservation_law_nats"]),
                "test/conservation_law_mc": float(split_metrics[curve_split]["mc_conservation_law_nats"]),
                "test/conservation_law_integral": float(split_metrics[curve_split]["integral_conservation_law_nats"]),
                "test/evaluated_tokens": int(split_metrics[curve_split]["evaluated_tokens"]),
                "test/evaluated_sequences": int(split_metrics[curve_split]["evaluated_sequences"]),
                "test/mse": float(split_metrics[curve_split]["mse"]),
            },
            step=int(cfg.train.n_steps),
        )
        logger.log_image("test/gexit_curve", output_dir / cfg.paths.curve_plot_filename, step=int(cfg.train.n_steps), caption=f"split={curve_split}")
        logger.log_directory_files(output_dir, artifact_name=f"{run_id}-outputs", exclude_dirs=[str(cfg.paths.wandb_dir_name)])
    barrier(dist_ctx)
    logger.finish()
    cleanup_distributed(dist_ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
