from __future__ import annotations

import sys
import time
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
    load_checkpoint_if_requested,
    load_history_prefix,
    load_img_benchmark,
    load_initial_weights_if_requested,
    maybe_apply_lm_lr_schedule,
    maybe_reset_peak_memory,
    resolve_output_dir,
    resolve_test_split_name,
    resolve_valid_split_name,
    runtime_metrics,
    save_discrete_reconstruction_panel,
    save_training_checkpoint,
)
from src.exit import apply_sequence_masking, per_token_nll
from src.img_models import RGB8ImageTokenUNet
from src.lm_distributed import all_reduce_mean_scalars, all_reduce_sum_scalars, barrier, cleanup_distributed, setup_distributed, unwrap_model
from src.lm_objectives import evaluate_exit_curve, evaluate_exit_objective_metrics, sample_mask_rates, save_exit_curve_plot


def _validate_cfg(cfg) -> None:
    if str(cfg.kind) != "img_exit":
        raise ValueError(f"Expected kind='img_exit', got {cfg.kind!r}.")
    if str(cfg.model.name).lower() != "img_unet_rgb8":
        raise ValueError("CIFAR image EXIT release configs require model.name='img_unet_rgb8'.")
    if str(cfg.benchmark.name).lower() != "cifar10_rgb8_tokens":
        raise ValueError("img_unet_rgb8 requires benchmark.name='cifar10_rgb8_tokens'.")
    if int(cfg.benchmark.context_length) != int(cfg.benchmark.image_size) ** 2 * 3:
        raise ValueError("cifar10_rgb8_tokens benchmark.context_length must equal image_size^2 * 3.")


def _build_model(cfg, *, vocab_size: int, context_length: int):
    return RGB8ImageTokenUNet(
        vocab_size=vocab_size,
        image_size=int(cfg.benchmark.image_size),
        base_channels=int(cfg.model.base_channels),
        channel_multipliers=tuple(int(value) for value in cfg.model.channel_multipliers),
        dropout=float(cfg.model.dropout),
        time_embed_input_dim=int(cfg.model.time_embed_input_dim),
        time_embed_dim=int(cfg.model.time_embed_dim),
        input_skip=bool(cfg.model.get("input_skip", False)),
    )


def _model_metadata(cfg, *, model_params: int) -> dict[str, object]:
    model_info: dict[str, object] = {
        "name": str(cfg.model.name),
        "parameter_count": int(model_params),
        "dropout": float(cfg.model.dropout),
        "base_channels": int(cfg.model.base_channels),
        "channel_multipliers": [int(value) for value in cfg.model.channel_multipliers],
        "time_embed_input_dim": int(cfg.model.time_embed_input_dim),
        "time_embed_dim": int(cfg.model.time_embed_dim),
        "input_skip": bool(cfg.model.get("input_skip", False)),
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
    resolved = None if checkpoint_path in (None, "") else (Path(str(checkpoint_path)).expanduser())
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

    logger = create_rank_aware_logger(
        cfg=cfg,
        run_id=run_id,
        output_dir=str(output_dir / cfg.paths.wandb_dir_name),
        enabled=dist_ctx.is_main_process,
    )
    model = _build_model(cfg, vocab_size=benchmark.vocab_size, context_length=benchmark.context_length).to(device)
    if cfg.train.resume_from not in (None, "") and cfg.train.get("init_weights_path") not in (None, ""):
        raise ValueError("Set only one of train.resume_from or train.init_weights_path for image EXIT runs.")
    initialized_from = load_initial_weights_if_requested(
        model=model,
        checkpoint_path=cfg.train.get("init_weights_path"),
        project_root=app_root,
        strict=bool(cfg.train.get("init_weights_strict", True)),
    )
    if initialized_from is not None and dist_ctx.is_main_process:
        print(f"[setup] initialized model weights from {initialized_from}")
    optimizer_name = str(cfg.train.get("optimizer", "adamw")).lower()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(cfg.train.lr),
            weight_decay=float(cfg.train.weight_decay),
            betas=(float(cfg.train.beta1), float(cfg.train.beta2)),
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg.train.lr),
            weight_decay=float(cfg.train.weight_decay),
            betas=(float(cfg.train.beta1), float(cfg.train.beta2)),
        )
    else:
        raise ValueError(f"Unsupported train.optimizer={optimizer_name!r}; expected 'adam' or 'adamw'.")
    start_step, tokens_seen = load_checkpoint_if_requested(
        model=model,
        optimizer=optimizer,
        checkpoint_path=cfg.train.resume_from,
        project_root=app_root,
    )
    if not bool(getattr(getattr(cfg.train, "lr_schedule", None), "enabled", False)):
        for param_group in optimizer.param_groups:
            param_group["lr"] = float(cfg.train.lr)
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
    train_generator.manual_seed(int(cfg.seed) + 303 + int(dist_ctx.rank) * 1_000)
    maybe_reset_peak_memory(device)
    start_time = time.perf_counter()
    valid_split = resolve_valid_split_name(benchmark)
    test_split = resolve_test_split_name(benchmark)
    train_mask_width = float(cfg.masking.train_mask_max) - float(cfg.masking.train_mask_min)
    valid_conservation_every = int(cfg.eval.eval_every)
    if valid_conservation_every <= 0:
        raise ValueError(f"eval.eval_every must be positive, got {valid_conservation_every}.")

    for step in range(start_step, int(cfg.train.n_steps) + 1):
        model.train()
        maybe_apply_lm_lr_schedule(cfg, step=step, optimizer=optimizer)
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        masked_ce_total = 0.0
        unmasked_ce_total = 0.0
        exit_gap_total = 0.0
        mask_rate_total = 0.0
        local_tokens_this_step = 0

        for _ in range(int(cfg.train.grad_accum_steps)):
            tokens = benchmark.sample_train_batch(batch_size=int(cfg.train.batch_size), device=device, include_next_token=False, generator=train_generator)
            s_per_sequence = sample_mask_rates(
                int(cfg.train.batch_size),
                min_rate=float(cfg.masking.train_mask_min),
                max_rate=float(cfg.masking.train_mask_max),
                device=device,
                generator=train_generator,
            )
            masked_tokens, mask, _ = apply_sequence_masking(tokens, s_per_sequence, mask_token_id=unwrap_model(model).mask_token_id, generator=train_generator)
            logits = model(masked_tokens, timesteps=s_per_sequence)
            losses = per_token_nll(logits, tokens)
            mask_f = mask.to(dtype=losses.dtype)
            unmasked_f = (~mask).to(dtype=losses.dtype)
            masked_count = torch.clamp(mask_f.sum(dim=1, keepdim=True), min=1.0)
            unmasked_count = torch.clamp(unmasked_f.sum(dim=1, keepdim=True), min=1.0)
            masked_ce = (torch.sum(losses * mask_f, dim=1, keepdim=True) / masked_count).mean()
            unmasked_ce = (torch.sum(losses * unmasked_f, dim=1, keepdim=True) / unmasked_count).mean()
            loss = masked_ce + unmasked_ce
            (loss / int(cfg.train.grad_accum_steps)).backward()
            loss_total += float(loss.item())
            masked_ce_total += float(masked_ce.item())
            unmasked_ce_total += float(unmasked_ce.item())
            exit_gap_total += float(masked_ce.item() - unmasked_ce.item())
            mask_rate_total += float(mask_f.mean().item())
            local_tokens_this_step += int(tokens.numel())

        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.train.max_grad_norm)).item())
        optimizer.step()
        if ema_state is not None:
            _update_ema_state(model, ema_state, decay=float(ema_decay))
        reduced_train_loss, reduced_masked_ce, reduced_unmasked_ce, reduced_exit_gap, reduced_mask_rate, reduced_grad_norm = all_reduce_mean_scalars(
            dist_ctx,
            [
                loss_total / int(cfg.train.grad_accum_steps),
                masked_ce_total / int(cfg.train.grad_accum_steps),
                unmasked_ce_total / int(cfg.train.grad_accum_steps),
                exit_gap_total / int(cfg.train.grad_accum_steps),
                mask_rate_total / int(cfg.train.grad_accum_steps),
                grad_norm,
            ],
            device=device,
        )
        global_tokens_this_step = int(all_reduce_sum_scalars(dist_ctx, [float(local_tokens_this_step)], device=device)[0])
        tokens_seen += int(global_tokens_this_step)
        row = {
            "step": int(step),
            "train_loss_nats": float(reduced_train_loss),
            "train_conservation_law_nats": float(train_mask_width * reduced_exit_gap),
            "masked_ce_nats": float(reduced_masked_ce),
            "unmasked_ce_nats": float(reduced_unmasked_ce),
            "exit_gap_nats": float(reduced_exit_gap),
            "empirical_mask_rate": float(reduced_mask_rate),
            "grad_norm": float(reduced_grad_norm),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "tokens_seen": int(tokens_seen),
            "epoch_progress": float(tokens_seen / max(train_token_count, 1)),
            "val_objective_loss_nats": None,
            "val_loss_nats": None,
            "val_masked_ce_nats": None,
            "val_unmasked_ce_nats": None,
            "val_exit_gap_nats": None,
            "val_empirical_mask_rate": None,
            "val_mean_s": None,
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
                    "train/masked_ce": row["masked_ce_nats"],
                    "train/unmasked_ce": row["unmasked_ce_nats"],
                    "train/grad_norm": row["grad_norm"],
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

        should_eval_valid = (
            eval_log_conservation_integral(cfg)
            and step % valid_conservation_every == 0
            and step < int(cfg.train.n_steps)
        )
        if should_eval_valid:
            val_metrics = evaluate_exit_objective_metrics(
                model, benchmark, split=valid_split, batch_size=int(cfg.eval.batch_size), device=device,
                min_rate=float(cfg.masking.train_mask_min), max_rate=float(cfg.masking.train_mask_max),
                max_batches=None if cfg.eval.max_batches is None else int(cfg.eval.max_batches),
                steps_per_batch=int(getattr(cfg.eval, "steps_per_batch", 1)),
                seed=int(cfg.seed) + 40_000 + step, dist_ctx=dist_ctx,
            )
            _valid_s_grid, _valid_curve, valid_curve_metrics, _valid_curve_rows = evaluate_exit_curve(
                model, benchmark, split=valid_split, batch_size=int(cfg.eval.batch_size), device=device,
                n_s_points=int(cfg.eval.n_s_points),
                max_batches=None if cfg.eval.max_batches is None else int(cfg.eval.max_batches),
                steps_per_batch=int(getattr(cfg.eval, "steps_per_batch", 1)),
                seed=int(cfg.seed) + 45_000 + step, dist_ctx=dist_ctx,
            )
            row.update(
                {
                    "val_objective_loss_nats": float(val_metrics["objective_loss_nats"]),
                    "val_loss_nats": float(val_metrics["loss_nats"]),
                    "val_masked_ce_nats": float(val_metrics["masked_ce_nats"]),
                    "val_unmasked_ce_nats": float(val_metrics["unmasked_ce_nats"]),
                    "val_exit_gap_nats": float(val_metrics["exit_gap_nats"]),
                    "val_empirical_mask_rate": float(val_metrics["empirical_mask_rate"]),
                    "val_mean_s": float(val_metrics["mean_s"]),
                    "val_evaluated_tokens": int(val_metrics["evaluated_tokens"]),
                    "val_evaluated_batches": int(val_metrics["evaluated_sequences"]),
                    "val_conservation_law_nats": float(valid_curve_metrics["curve_integral_nats"]),
                    "val_conservation_law_mc_nats": float(val_metrics["mc_conservation_law_nats"]),
                    "val_conservation_law_integral_nats": float(valid_curve_metrics["curve_integral_nats"]),
                }
            )
            row["valid_conservation_law_nats"] = row["val_conservation_law_nats"]
            logger.log(
                {
                    "valid/loss": row["val_loss_nats"],
                    "valid/conservation_law": row["val_conservation_law_nats"],
                    "valid/conservation_law_mc": row["val_conservation_law_mc_nats"],
                    "valid/conservation_law_integral": row["val_conservation_law_integral_nats"],
                    "valid/evaluated_tokens": row["val_evaluated_tokens"],
                    "valid/evaluated_sequences": row["val_evaluated_batches"],
                    "valid/masked_ce": row["val_masked_ce_nats"],
                    "valid/unmasked_ce": row["val_unmasked_ce_nats"],
                },
                step=step,
            )

    if not eval_generate_curves(cfg):
        if dist_ctx.is_main_process:
            save_csv(output_dir / cfg.paths.history_filename, history_rows)
        barrier(dist_ctx)
        logger.finish()
        cleanup_distributed(dist_ctx)
        return 0

    curve_split = test_split
    split_metrics: dict[str, dict[str, float]] = {}
    primary_curve_rows: list[dict[str, float]] = []
    primary_curve = None
    primary_grid = None
    for index, split in enumerate([valid_split] if valid_split == test_split else [valid_split, test_split]):
        split_name = str(split)
        objective_metrics = evaluate_exit_objective_metrics(
            model, benchmark, split=split_name, batch_size=int(cfg.eval.batch_size), device=device,
            min_rate=float(cfg.masking.train_mask_min), max_rate=float(cfg.masking.train_mask_max),
            max_batches=None if cfg.eval.max_batches is None else int(cfg.eval.max_batches),
            steps_per_batch=int(getattr(cfg.eval, "steps_per_batch", 1)),
            seed=int(cfg.seed) + 5_000 + index * 1000, dist_ctx=dist_ctx,
        )
        s_grid, curve_dict, metrics, curve_rows = evaluate_exit_curve(
            model, benchmark, split=split_name, batch_size=int(cfg.eval.batch_size), device=device,
            n_s_points=int(cfg.eval.n_s_points),
            max_batches=None if cfg.eval.max_batches is None else int(cfg.eval.max_batches),
            steps_per_batch=int(getattr(cfg.eval, "steps_per_batch", 1)),
            seed=int(cfg.seed) + 10_000 + index * 1000, dist_ctx=dist_ctx,
        )
        split_metrics[split_name] = {**objective_metrics, **metrics}
        split_metrics[split_name]["mc_conservation_law_nats"] = float(objective_metrics["mc_conservation_law_nats"])
        split_metrics[split_name]["integral_conservation_law_nats"] = float(split_metrics[split_name]["curve_integral_nats"])
        split_metrics[split_name]["conservation_law_nats"] = conservation_law_nats(objective=str(cfg.objective), metrics=split_metrics[split_name])
        if split_name == curve_split:
            primary_curve_rows = curve_rows
            primary_curve = curve_dict
            primary_grid = s_grid
    assert primary_curve is not None and primary_grid is not None
    attach_bpd_to_split_metrics(benchmark=benchmark, objective=str(cfg.objective), split_metrics=split_metrics)
    metrics = {
        "run_id": run_id,
        "experiment_kind": str(cfg.kind),
        "objective": str(cfg.objective),
        "benchmark": benchmark.benchmark_metadata(),
        "model": _model_metadata(cfg, model_params=model_params),
        "train": {
            "n_steps": int(cfg.train.n_steps),
            "batch_size": int(cfg.train.batch_size),
            "grad_accum_steps": int(cfg.train.grad_accum_steps),
            "lr": float(cfg.train.lr),
            "weight_decay": float(cfg.train.weight_decay),
            "max_grad_norm": float(cfg.train.max_grad_norm),
            "tokens_seen": int(tokens_seen),
            "epoch_progress": float(tokens_seen / max(train_token_count, 1)),
            "train_mask_min": float(cfg.masking.train_mask_min),
            "train_mask_max": float(cfg.masking.train_mask_max),
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
        save_exit_curve_plot(output_dir / cfg.paths.curve_plot_filename, s_grid=primary_grid, exit_curve=primary_curve["exit_gap"], title=f"{benchmark.name} gEXIT curve ({curve_split}, masking noise)", area=float(split_metrics[curve_split]["curve_integral_nats"]))
        sample_batch = next(benchmark.iter_eval_batches(split=curve_split, batch_size=4, device=device, include_next_token=False, max_batches=1))
        clean_images = benchmark.split_images(curve_split)
        clean_images = None if clean_images is None else clean_images[: int(sample_batch.shape[0])]
        panel_s = torch.full((int(sample_batch.shape[0]),), 0.5, dtype=torch.float32, device=device)
        observed_tokens, _, _ = apply_sequence_masking(sample_batch, panel_s, mask_token_id=unwrap_model(model).mask_token_id)
        predicted_tokens = torch.argmax(model(observed_tokens, timesteps=panel_s), dim=-1)
        save_discrete_reconstruction_panel(benchmark, clean_tokens=sample_batch.cpu(), observed_tokens=observed_tokens.cpu(), predicted_tokens=predicted_tokens.cpu(), output_path=output_dir / cfg.paths.reconstruction_plot_filename, mask_token_id=unwrap_model(model).mask_token_id, clean_images=clean_images)
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
                "test/masked_ce": float(split_metrics[curve_split]["masked_ce_nats"]),
                "test/unmasked_ce": float(split_metrics[curve_split]["unmasked_ce_nats"]),
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
