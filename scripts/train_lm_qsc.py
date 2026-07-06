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
from scripts.lm_common import (
    conservation_law_nats,
    compose_lm_config,
    count_parameters,
    create_rank_aware_logger,
    eval_generate_curves,
    eval_log_conservation_integral,
    find_app_root,
    find_repo_root,
    lm_embedding_norm,
    load_checkpoint_if_requested,
    load_history_prefix,
    load_initial_weights_if_requested,
    maybe_apply_lm_lr_schedule,
    maybe_reset_peak_memory,
    resolve_test_split_name,
    resolve_valid_split_name,
    resolve_output_dir,
    runtime_metrics,
    save_training_checkpoint,
)
from src.lm_data import load_lm_benchmark
from src.lm_distributed import (
    all_reduce_mean_scalars,
    all_reduce_sum_scalars,
    barrier,
    cleanup_distributed,
    setup_distributed,
)
from src.lm_models import MaskedTransformerLM
from src.lm_objectives import (
    compute_qsc_loss,
    evaluate_qsc_curve,
    evaluate_qsc_objective_metrics,
    save_qsc_curve_plot,
    sample_mask_rates,
)


def _validate_cfg(cfg) -> None:
    if str(cfg.kind) != "lm_qsc":
        raise ValueError(f"Expected kind='lm_qsc', got {cfg.kind!r}.")
    if not 0.0 < float(cfg.qsc.train_t_min) < float(cfg.qsc.train_t_max) < 1.0:
        raise ValueError("QSC training t range must satisfy 0 < min < max < 1.")
    if not 0.0 < float(cfg.eval.t_min) < float(cfg.eval.t_max) < 1.0:
        raise ValueError("QSC eval t range must satisfy 0 < min < max < 1.")


def _build_model(cfg, *, vocab_size: int, context_length: int) -> MaskedTransformerLM:
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


def main(argv: list[str] | None = None) -> int:
    configure_environment()
    argv = sys.argv[1:] if argv is None else argv
    app_root = find_app_root(Path(__file__).resolve())
    repo_root = find_repo_root(app_root)
    config_name, overrides, cfg = compose_lm_config(project_root=app_root, argv=argv)
    _validate_cfg(cfg)

    dist_ctx, device = setup_distributed(str(cfg.device))
    set_global_seed(int(cfg.seed) + int(dist_ctx.rank), deterministic=bool(cfg.deterministic))
    benchmark = load_lm_benchmark(cfg.benchmark, project_root=repo_root)
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
        raise ValueError("Set only one of train.resume_from or train.init_weights_path for LM QSC runs.")
    initialized_from = load_initial_weights_if_requested(
        model=model,
        checkpoint_path=cfg.train.get("init_weights_path"),
        project_root=app_root,
        strict=bool(cfg.train.get("init_weights_strict", True)),
    )
    if initialized_from is not None and dist_ctx.is_main_process:
        print(f"[setup] initialized model weights from {initialized_from}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
        betas=(float(cfg.train.beta1), float(cfg.train.beta2)),
    )
    start_step, tokens_seen = load_checkpoint_if_requested(
        model=model,
        optimizer=optimizer,
        checkpoint_path=cfg.train.resume_from,
        project_root=app_root,
    )
    if dist_ctx.enabled:
        model = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
        )
    model_params = count_parameters(model)
    logger.log({"model/parameter_count": int(model_params)})

    history_rows: list[dict[str, float | int | str]] = load_history_prefix(
        output_dir / cfg.paths.history_filename,
        start_step=start_step,
    )
    train_generator = torch.Generator(device="cpu")
    train_generator.manual_seed(int(cfg.seed) + 909 + int(dist_ctx.rank) * 1_000)
    maybe_reset_peak_memory(device)
    start_time = time.perf_counter()
    valid_split = resolve_valid_split_name(benchmark)
    test_split = resolve_test_split_name(benchmark)
    train_t_width = float(cfg.qsc.train_t_max) - float(cfg.qsc.train_t_min)
    valid_conservation_every = int(cfg.eval.eval_every)
    if valid_conservation_every <= 0:
        raise ValueError(f"eval.eval_every must be positive, got {valid_conservation_every}.")

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
            tokens = benchmark.sample_train_batch(
                batch_size=int(cfg.train.batch_size),
                device=device,
                include_next_token=False,
                generator=train_generator,
            )
            t_per_sequence = sample_mask_rates(
                int(cfg.train.batch_size),
                min_rate=float(cfg.qsc.train_t_min),
                max_rate=float(cfg.qsc.train_t_max),
                device=device,
                generator=train_generator,
            )
            stats = compute_qsc_loss(
                model,
                tokens=tokens,
                t=t_per_sequence,
                generator=train_generator,
                label_smoothing=float(cfg.train.label_smoothing),
            )
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
        row = {
            "step": int(step),
            "train_loss_nats": float(reduced_loss),
            "train_conservation_law_nats": float(train_t_width * reduced_qsc_gexit),
            "train_ce_correct_nats": float(reduced_correct_ce),
            "train_ce_incorrect_nats": float(reduced_incorrect_ce),
            "train_posterior_mismatched_nats": float(reduced_posterior_mismatched),
            "train_qsc_gexit_nats": float(reduced_qsc_gexit),
            "grad_norm": float(reduced_grad_norm),
            "embed_norm": lm_embedding_norm(model),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "tokens_seen": int(tokens_seen),
            "epoch_progress": float(tokens_seen / max(train_token_count, 1)),
            "val_objective_loss_nats": None,
            "val_loss_nats": None,
            "val_ce_correct_nats": None,
            "val_ce_incorrect_nats": None,
            "val_posterior_mismatched_nats": None,
            "val_qsc_gexit_nats": None,
            "val_qsc_gexit_mc_nats": None,
            "val_qsc_gexit_integral_nats": None,
            "val_mean_t": None,
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
                    "train/clean_ce": row["train_ce_correct_nats"],
                    "train/substitute_ce": row["train_ce_incorrect_nats"],
                    "train/posterior_mismatched": row["train_posterior_mismatched_nats"],
                    "train/grad_norm": row["grad_norm"],
                    "train/embed_norm": row["embed_norm"],
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
            )

        should_eval_valid = (
            eval_log_conservation_integral(cfg)
            and step % valid_conservation_every == 0
            and step < int(cfg.train.n_steps)
        )
        if should_eval_valid:
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
                seed=int(cfg.seed) + 60_000 + step,
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
                seed=int(cfg.seed) + 70_000 + step,
                dist_ctx=dist_ctx,
            )
            valid_t_width = float(cfg.eval.t_max) - float(cfg.eval.t_min)
            valid_qsc_gexit_integral = float(valid_curve_metrics["curve_integral_nats"]) / valid_t_width
            row.update(
                {
                    "val_objective_loss_nats": float(val_metrics["objective_loss_nats"]),
                    "val_loss_nats": float(val_metrics["loss_nats"]),
                    "val_ce_correct_nats": float(val_metrics["ce_correct_nats"]),
                    "val_ce_incorrect_nats": float(val_metrics["ce_incorrect_nats"]),
                    "val_posterior_mismatched_nats": float(val_metrics["posterior_mismatched"]),
                    "val_qsc_gexit_nats": valid_qsc_gexit_integral,
                    "val_qsc_gexit_mc_nats": float(val_metrics["qsc_gexit_nats"]),
                    "val_qsc_gexit_integral_nats": valid_qsc_gexit_integral,
                    "val_mean_t": float(val_metrics["mean_t"]),
                    "val_evaluated_tokens": int(val_metrics["evaluated_tokens"]),
                    "val_evaluated_batches": int(val_metrics["evaluated_sequences"]),
                }
            )
            row["val_conservation_law_nats"] = float(valid_curve_metrics["curve_integral_nats"])
            row["val_conservation_law_mc_nats"] = float(val_metrics["conservation_law_nats"])
            row["val_conservation_law_integral_nats"] = row["val_conservation_law_nats"]
            row["valid_conservation_law_nats"] = row["val_conservation_law_nats"]
            logger.log(
                {
                    "valid/loss": row["val_loss_nats"],
                    "valid/conservation_law": row["val_conservation_law_nats"],
                    "valid/conservation_law_mc": row["val_conservation_law_mc_nats"],
                    "valid/conservation_law_integral": row["val_conservation_law_integral_nats"],
                    "valid/evaluated_tokens": row["val_evaluated_tokens"],
                    "valid/evaluated_sequences": row["val_evaluated_batches"],
                    "valid/clean_ce": row["val_ce_correct_nats"],
                    "valid/substitute_ce": row["val_ce_incorrect_nats"],
                    "valid/posterior_mismatched": row["val_posterior_mismatched_nats"],
                    "valid/qsc_gexit": row["val_qsc_gexit_nats"],
                    "valid/qsc_gexit_mc": row["val_qsc_gexit_mc_nats"],
                    "valid/qsc_gexit_integral": row["val_qsc_gexit_integral_nats"],
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
        objective_metrics = evaluate_qsc_objective_metrics(
            model,
            benchmark,
            split=split_name,
            batch_size=int(cfg.eval.batch_size),
            device=device,
            t_min=float(cfg.eval.t_min),
            t_max=float(cfg.eval.t_max),
            max_batches=None if cfg.eval.max_batches is None else int(cfg.eval.max_batches),
            steps_per_batch=int(getattr(cfg.eval, "steps_per_batch", 1)),
            seed=int(cfg.seed) + 5_000 + index * 1000,
            dist_ctx=dist_ctx,
        )
        t_grid, curve_dict, metrics, curve_rows = evaluate_qsc_curve(
            model,
            benchmark,
            split=split_name,
            batch_size=int(cfg.eval.batch_size),
            device=device,
            t_min=float(cfg.eval.t_min),
            t_max=float(cfg.eval.t_max),
            n_t_points=int(cfg.eval.n_t_points),
            max_batches=None if cfg.eval.max_batches is None else int(cfg.eval.max_batches),
            steps_per_batch=int(getattr(cfg.eval, "steps_per_batch", 1)),
            seed=int(cfg.seed) + 10_000 + index * 1000,
            dist_ctx=dist_ctx,
        )
        split_metrics[split_name] = {**objective_metrics, **metrics}
        t_width = float(cfg.eval.t_max) - float(cfg.eval.t_min)
        split_metrics[split_name]["mc_conservation_law_nats"] = float(objective_metrics["conservation_law_nats"])
        split_metrics[split_name]["integral_conservation_law_nats"] = float(split_metrics[split_name]["curve_integral_nats"])
        split_metrics[split_name]["qsc_gexit_mc_nats"] = float(objective_metrics["qsc_gexit_nats"])
        split_metrics[split_name]["qsc_gexit_integral_nats"] = float(split_metrics[split_name]["curve_integral_nats"]) / t_width
        split_metrics[split_name]["qsc_gexit_nats"] = float(split_metrics[split_name]["curve_integral_nats"]) / t_width
        split_metrics[split_name]["conservation_law_nats"] = conservation_law_nats(
            objective=str(cfg.objective),
            metrics=split_metrics[split_name],
        )
        if split_name == curve_split:
            primary_curve_rows = curve_rows
            primary_curve = curve_dict
            primary_grid = t_grid

    assert primary_curve is not None and primary_grid is not None

    metrics = {
        "run_id": run_id,
        "experiment_kind": str(cfg.kind),
        "objective": str(cfg.objective),
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
            "tokens_seen": int(tokens_seen),
            "epoch_progress": float(tokens_seen / max(train_token_count, 1)),
            "train_t_min": float(cfg.qsc.train_t_min),
            "train_t_max": float(cfg.qsc.train_t_max),
        },
        "nats_per_token": {split: float(values["nats_per_token"]) for split, values in split_metrics.items()},
        "eval_details": split_metrics,
        "ranking_metric_nats_per_token": float(split_metrics[curve_split]["nats_per_token"]),
        "primary_split": curve_split,
        **runtime_metrics(start_time=start_time, device=device),
    }
    if dist_ctx.is_main_process:
        save_csv(output_dir / cfg.paths.curve_filename, primary_curve_rows)
        save_qsc_curve_plot(
            output_dir / cfg.paths.curve_plot_filename,
            t_grid=primary_grid,
            qsc_curve=primary_curve["qsc_gexit"],
            title=f"{benchmark.name} gEXIT curve ({curve_split}, q-ary substitution noise)",
            area=float(split_metrics[curve_split]["curve_integral_nats"]),
        )
        save_csv(output_dir / cfg.paths.history_filename, history_rows)
        save_json(output_dir / cfg.paths.metrics_filename, metrics)
        torch.save(model.module.state_dict() if hasattr(model, "module") else model.state_dict(), output_dir / cfg.paths.model_filename)
        logger.log(
            {
                "test/loss": float(split_metrics[curve_split]["loss_nats"]),
                "test/conservation_law": float(split_metrics[curve_split]["conservation_law_nats"]),
                "test/evaluated_tokens": int(split_metrics[curve_split]["evaluated_tokens"]),
                "test/evaluated_sequences": int(split_metrics[curve_split]["evaluated_sequences"]),
                "test/clean_ce": float(split_metrics[curve_split]["clean_ce"]),
                "test/substitute_ce": float(split_metrics[curve_split]["substitute_ce"]),
                "test/posterior_mismatched": float(split_metrics[curve_split]["posterior_mismatched"]),
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
