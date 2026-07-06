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
from src.img_models import RGB8ImageTokenUNet
from src.lm_distributed import all_reduce_mean_scalars, all_reduce_sum_scalars, barrier, cleanup_distributed, setup_distributed
from src.lm_objectives import compute_qsc_loss, evaluate_qsc_curve, evaluate_qsc_objective_metrics, sample_mask_rates, save_qsc_curve_plot


def _validate_cfg(cfg) -> None:
    if str(cfg.kind) != "img_qsc":
        raise ValueError(f"Expected kind='img_qsc', got {cfg.kind!r}.")
    if str(cfg.model.name).lower() != "img_unet_rgb8":
        raise ValueError("CIFAR image QSC release configs require model.name='img_unet_rgb8'.")
    if str(cfg.benchmark.name).lower() != "cifar10_rgb8_tokens":
        raise ValueError("img_unet_rgb8 requires benchmark.name='cifar10_rgb8_tokens'.")


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
    initialized_from = load_initial_weights_if_requested(model=model, checkpoint_path=cfg.train.get("init_weights_path"), project_root=app_root, strict=bool(cfg.train.get("init_weights_strict", True)))
    if initialized_from is not None and dist_ctx.is_main_process:
        print(f"[setup] initialized model weights from {initialized_from}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay), betas=(float(cfg.train.beta1), float(cfg.train.beta2)))
    start_step, tokens_seen = load_checkpoint_if_requested(model=model, optimizer=optimizer, checkpoint_path=cfg.train.resume_from, project_root=app_root)
    if dist_ctx.enabled:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None, output_device=device.index if device.type == "cuda" else None)
    model_params = count_parameters(model)
    logger.log({"model/parameter_count": int(model_params)})
    history_rows = load_history_prefix(output_dir / cfg.paths.history_filename, start_step=start_step)
    train_generator = torch.Generator(device="cpu")
    train_generator.manual_seed(int(cfg.seed) + 909 + int(dist_ctx.rank) * 1_000)
    maybe_reset_peak_memory(device)
    start_time = time.perf_counter()
    valid_split = resolve_valid_split_name(benchmark)
    test_split = resolve_test_split_name(benchmark)
    train_t_width = float(cfg.qsc.train_t_max) - float(cfg.qsc.train_t_min)
    save_every = None if cfg.train.get("save_every") is None else max(1, int(cfg.train.save_every))
    eval_every = None if cfg.eval.get("eval_every") is None else max(1, int(cfg.eval.eval_every))
    valid_conservation_every = eval_every if eval_every is not None else 5_000

    for step in range(start_step, int(cfg.train.n_steps) + 1):
        model.train()
        maybe_apply_lm_lr_schedule(cfg, step=step, optimizer=optimizer)
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        correct_ce_total = 0.0
        incorrect_ce_total = 0.0
        posterior_mismatched_total = 0.0
        qsc_gexit_total = 0.0
        local_tokens_this_step = 0
        for _ in range(int(cfg.train.grad_accum_steps)):
            tokens = benchmark.sample_train_batch(batch_size=int(cfg.train.batch_size), device=device, include_next_token=False, generator=train_generator)
            t_per_sequence = sample_mask_rates(int(cfg.train.batch_size), min_rate=float(cfg.qsc.train_t_min), max_rate=float(cfg.qsc.train_t_max), device=device, generator=train_generator)
            stats = compute_qsc_loss(model, tokens=tokens, t=t_per_sequence, generator=train_generator, label_smoothing=float(cfg.train.label_smoothing))
            (stats["loss"] / int(cfg.train.grad_accum_steps)).backward()
            loss_total += float(stats["loss"].item())
            correct_ce_total += float(stats["correct_ce"].item())
            incorrect_ce_total += float(stats["incorrect_ce"].item())
            posterior_mismatched_total += float(stats["posterior_mismatched"].item())
            qsc_gexit_total += float(stats["qsc_gexit"].item())
            local_tokens_this_step += int(tokens.numel())
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.train.max_grad_norm)).item())
        optimizer.step()
        reduced_loss, reduced_correct_ce, reduced_incorrect_ce, reduced_posterior_mismatched, reduced_qsc_gexit, reduced_grad_norm = all_reduce_mean_scalars(
            dist_ctx,
            [
                loss_total / int(cfg.train.grad_accum_steps),
                correct_ce_total / int(cfg.train.grad_accum_steps),
                incorrect_ce_total / int(cfg.train.grad_accum_steps),
                posterior_mismatched_total / int(cfg.train.grad_accum_steps),
                qsc_gexit_total / int(cfg.train.grad_accum_steps),
                grad_norm,
            ],
            device=device,
        )
        global_tokens_this_step = int(all_reduce_sum_scalars(dist_ctx, [float(local_tokens_this_step)], device=device)[0])
        tokens_seen += int(global_tokens_this_step)
        history_rows.append(
            {
                "step": int(step),
                "train_loss_nats": float(reduced_loss),
                "train_conservation_law_nats": float(train_t_width * reduced_qsc_gexit),
                "train_ce_correct_nats": float(reduced_correct_ce),
                "train_ce_incorrect_nats": float(reduced_incorrect_ce),
                "train_posterior_mismatched_nats": float(reduced_posterior_mismatched),
                "train_qsc_gexit_nats": float(reduced_qsc_gexit),
                "grad_norm": float(reduced_grad_norm),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "tokens_seen": int(tokens_seen),
                "epoch_progress": float(tokens_seen / max(train_token_count, 1)),
                "valid_conservation_law_nats": None,
                "valid_conservation_law_mc_nats": None,
                "valid_conservation_law_integral_nats": None,
            }
        )
        row = history_rows[-1]
        if step == start_step or step % int(cfg.logging.log_every) == 0 or step == int(cfg.train.n_steps):
            logger.log(
                {
                    "train/loss": row["train_loss_nats"],
                    "train/conservation_law": row["train_conservation_law_nats"],
                    "train/clean_ce": row["train_ce_correct_nats"],
                    "train/substitute_ce": row["train_ce_incorrect_nats"],
                    "train/posterior_mismatched": row["train_posterior_mismatched_nats"],
                    "train/grad_norm": row["grad_norm"],
                    "train/lr": row["lr"],
                    "train/tokens_seen": row["tokens_seen"],
                    "train/epoch_progress": row["epoch_progress"],
                },
                step=step,
            )
        if save_every is not None and dist_ctx.is_main_process and step > 0 and step % save_every == 0 and step < int(cfg.train.n_steps):
            save_training_checkpoint(
                output_dir=output_dir,
                checkpoint_dir_name=str(cfg.paths.checkpoint_dir_name),
                filename_template=str(cfg.paths.checkpoint_filename_template),
                step=int(step),
                model=model,
                optimizer=optimizer,
                tokens_seen=int(tokens_seen),
            )
            save_csv(output_dir / cfg.paths.history_filename, history_rows)
        if eval_log_conservation_integral(cfg) and step % valid_conservation_every == 0 and step < int(cfg.train.n_steps):
            val_metrics = evaluate_qsc_objective_metrics(
                model,
                benchmark,
                split=valid_split,
                batch_size=int(cfg.eval.batch_size),
                device=device,
                t_min=float(cfg.eval.t_min),
                t_max=float(cfg.eval.t_max),
                max_batches=None if cfg.eval.max_batches is None else int(cfg.eval.max_batches),
                steps_per_batch=int(getattr(cfg.eval, "steps_per_batch", 1)),
                seed=int(cfg.seed) + 6_500 + step,
                dist_ctx=dist_ctx,
            )
            _valid_t_grid, _valid_curve, valid_curve_metrics, _valid_curve_rows = evaluate_qsc_curve(
                model,
                benchmark,
                split=valid_split,
                batch_size=int(cfg.eval.batch_size),
                device=device,
                t_min=float(cfg.eval.t_min),
                t_max=float(cfg.eval.t_max),
                n_t_points=int(cfg.eval.n_t_points),
                max_batches=None if cfg.eval.max_batches is None else int(cfg.eval.max_batches),
                steps_per_batch=int(getattr(cfg.eval, "steps_per_batch", 1)),
                seed=int(cfg.seed) + 7_500 + step,
                dist_ctx=dist_ctx,
            )
            row["valid_conservation_law_nats"] = float(valid_curve_metrics["curve_integral_nats"])
            row["valid_conservation_law_mc_nats"] = float(val_metrics["conservation_law_nats"])
            row["valid_conservation_law_integral_nats"] = row["valid_conservation_law_nats"]
            logger.log(
                {
                    "valid/loss": float(val_metrics["loss_nats"]),
                    "valid/conservation_law": row["valid_conservation_law_nats"],
                    "valid/conservation_law_mc": row["valid_conservation_law_mc_nats"],
                    "valid/conservation_law_integral": row["valid_conservation_law_integral_nats"],
                    "valid/evaluated_tokens": int(val_metrics["evaluated_tokens"]),
                    "valid/evaluated_sequences": int(val_metrics["evaluated_sequences"]),
                    "valid/clean_ce": float(val_metrics["clean_ce"]),
                    "valid/substitute_ce": float(val_metrics["substitute_ce"]),
                    "valid/posterior_mismatched": float(val_metrics["posterior_mismatched"]),
                },
                step=step,
            )
    split_metrics: dict[str, dict[str, float]] = {}
    primary_curve_rows = []
    primary_curve = None
    primary_grid = None
    for index, split in enumerate([valid_split] if valid_split == test_split else [valid_split, test_split]):
        split_name = str(split)
        objective_metrics = evaluate_qsc_objective_metrics(
            model, benchmark, split=split_name, batch_size=int(cfg.eval.batch_size), device=device, t_min=float(cfg.eval.t_min), t_max=float(cfg.eval.t_max),
            max_batches=None if cfg.eval.max_batches is None else int(cfg.eval.max_batches), steps_per_batch=int(getattr(cfg.eval, "steps_per_batch", 1)), seed=int(cfg.seed) + 5_000 + index * 1000, dist_ctx=dist_ctx,
        )
        t_grid, curve_dict, metrics, curve_rows = evaluate_qsc_curve(
            model, benchmark, split=split_name, batch_size=int(cfg.eval.batch_size), device=device, t_min=float(cfg.eval.t_min), t_max=float(cfg.eval.t_max),
            n_t_points=int(cfg.eval.n_t_points), max_batches=None if cfg.eval.max_batches is None else int(cfg.eval.max_batches), steps_per_batch=int(getattr(cfg.eval, "steps_per_batch", 1)), seed=int(cfg.seed) + 10_000 + index * 1000, dist_ctx=dist_ctx,
        )
        split_metrics[split_name] = {**objective_metrics, **metrics}
        t_width = float(cfg.eval.t_max) - float(cfg.eval.t_min)
        split_metrics[split_name]["mc_conservation_law_nats"] = float(objective_metrics["conservation_law_nats"])
        split_metrics[split_name]["integral_conservation_law_nats"] = float(split_metrics[split_name]["curve_integral_nats"])
        split_metrics[split_name]["qsc_gexit_mc_nats"] = float(objective_metrics["qsc_gexit_nats"])
        split_metrics[split_name]["qsc_gexit_integral_nats"] = float(split_metrics[split_name]["curve_integral_nats"]) / t_width
        split_metrics[split_name]["qsc_gexit_nats"] = float(split_metrics[split_name]["qsc_gexit_integral_nats"])
        split_metrics[split_name]["conservation_law_nats"] = conservation_law_nats(objective=str(cfg.objective), metrics=split_metrics[split_name])
        if split_name == test_split:
            primary_curve_rows = curve_rows
            primary_curve = curve_dict
            primary_grid = t_grid
    assert primary_curve is not None and primary_grid is not None
    attach_bpd_to_split_metrics(benchmark=benchmark, objective=str(cfg.objective), split_metrics=split_metrics)
    metrics = {
        "run_id": run_id,
        "experiment_kind": str(cfg.kind),
        "objective": str(cfg.objective),
        "benchmark": benchmark.benchmark_metadata(),
        "model": _model_metadata(cfg, model_params=model_params),
        "train": {"n_steps": int(cfg.train.n_steps), "batch_size": int(cfg.train.batch_size), "grad_accum_steps": int(cfg.train.grad_accum_steps), "lr": float(cfg.train.lr), "tokens_seen": int(tokens_seen)},
        "nats_per_token": {split: float(values["nats_per_token"]) for split, values in split_metrics.items()},
        "bpd": {split: float(values["bpd"]) for split, values in split_metrics.items() if "bpd" in values},
        "eval_details": split_metrics,
        "ranking_metric_nats_per_token": float(split_metrics[test_split]["nats_per_token"]),
        "ranking_metric_bpd": float(split_metrics[test_split]["bpd"]),
        "primary_split": test_split,
        **runtime_metrics(start_time=start_time, device=device),
    }
    if dist_ctx.is_main_process:
        save_csv(output_dir / cfg.paths.curve_filename, primary_curve_rows)
        save_qsc_curve_plot(output_dir / cfg.paths.curve_plot_filename, t_grid=primary_grid, qsc_curve=primary_curve["qsc_gexit"], title=f"{benchmark.name} gEXIT curve ({test_split}, q-ary substitution noise)", area=float(split_metrics[test_split]["curve_integral_nats"]))
        sample_batch = next(benchmark.iter_eval_batches(split=test_split, batch_size=4, device=device, include_next_token=False, max_batches=1))
        clean_images = benchmark.split_images(test_split)
        clean_images = None if clean_images is None else clean_images[: int(sample_batch.shape[0])]
        t_per_sequence = torch.full((int(sample_batch.shape[0]),), 0.5, dtype=torch.float32, device=device)
        stats = compute_qsc_loss(model, tokens=sample_batch, t=t_per_sequence)
        predicted_tokens = torch.argmax(stats["logits_reweighted"], dim=-1)
        save_discrete_reconstruction_panel(benchmark, clean_tokens=sample_batch.cpu(), observed_tokens=stats["observed_tokens"].cpu(), predicted_tokens=predicted_tokens.cpu(), output_path=output_dir / cfg.paths.reconstruction_plot_filename, clean_images=clean_images)
        save_csv(output_dir / cfg.paths.history_filename, history_rows)
        save_json(output_dir / cfg.paths.metrics_filename, metrics)
        torch.save(model.module.state_dict() if hasattr(model, "module") else model.state_dict(), output_dir / cfg.paths.model_filename)
        logger.log(
            {
                "test/loss": float(split_metrics[test_split]["loss_nats"]),
                "test/conservation_law": float(split_metrics[test_split]["conservation_law_nats"]),
                "test/conservation_law_mc": float(split_metrics[test_split]["mc_conservation_law_nats"]),
                "test/conservation_law_integral": float(split_metrics[test_split]["integral_conservation_law_nats"]),
                "test/evaluated_tokens": int(split_metrics[test_split]["evaluated_tokens"]),
                "test/evaluated_sequences": int(split_metrics[test_split]["evaluated_sequences"]),
                "test/clean_ce": float(split_metrics[test_split]["clean_ce"]),
                "test/substitute_ce": float(split_metrics[test_split]["substitute_ce"]),
                "test/posterior_mismatched": float(split_metrics[test_split]["posterior_mismatched"]),
            },
            step=int(cfg.train.n_steps),
        )
        logger.log_image("test/gexit_curve", output_dir / cfg.paths.curve_plot_filename, step=int(cfg.train.n_steps), caption=f"split={test_split}")
        logger.log_directory_files(output_dir, artifact_name=f"{run_id}-outputs", exclude_dirs=[str(cfg.paths.wandb_dir_name)])
    barrier(dist_ctx)
    logger.finish()
    cleanup_distributed(dist_ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
