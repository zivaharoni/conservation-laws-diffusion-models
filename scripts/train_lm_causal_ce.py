from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
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
    load_history_prefix,
    load_checkpoint_if_requested,
    maybe_apply_lm_lr_schedule,
    maybe_reset_peak_memory,
    prefixed_scalar_metrics,
    resolve_output_dir,
    save_training_checkpoint,
    runtime_metrics,
)
from src.lm_data import load_lm_benchmark
from src.lm_distributed import all_reduce_mean_scalars, all_reduce_sum_scalars, barrier, cleanup_distributed, setup_distributed
from src.lm_models import CausalTransformerLM
from src.lm_objectives import evaluate_causal_nats


def _validate_cfg(cfg) -> None:
    if str(cfg.kind) != "lm_causal_ce":
        raise ValueError(f"Expected kind='lm_causal_ce', got {cfg.kind!r}.")
    if int(cfg.benchmark.context_length) <= 1:
        raise ValueError("benchmark.context_length must be at least 2 for causal next-token training.")
    if int(cfg.train.batch_size) <= 0 or int(cfg.train.grad_accum_steps) <= 0:
        raise ValueError("train.batch_size and train.grad_accum_steps must be positive.")
    if int(cfg.train.n_steps) <= 0:
        raise ValueError("train.n_steps must be positive.")


def _build_model(cfg, *, vocab_size: int, context_length: int) -> CausalTransformerLM:
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
    if cfg.verbose and dist_ctx.is_main_process:
        print(f"[setup] benchmark={benchmark.name} run_id={run_id} device={device} output_dir={output_dir}")
        print(f"[setup] tokenizer={benchmark.tokenizer.name} vocab_size={benchmark.vocab_size}")
        print(f"[setup] context_length={benchmark.context_length}")
    if dist_ctx.is_main_process:
        save_config(output_dir / cfg.paths.config_snapshot_name, cfg)

    logger = create_rank_aware_logger(
        cfg=cfg,
        run_id=run_id,
        output_dir=str(output_dir / cfg.paths.wandb_dir_name),
        enabled=dist_ctx.is_main_process,
    )
    model = _build_model(cfg, vocab_size=benchmark.vocab_size, context_length=benchmark.context_length).to(device)
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
    train_generator.manual_seed(int(cfg.seed) + 101 + int(dist_ctx.rank) * 1_000)
    maybe_reset_peak_memory(device)
    start_time = time.perf_counter()
    eval_every = int(cfg.eval.eval_every) if cfg.eval.eval_every is not None else 0
    primary_split = str(cfg.eval.primary_split)

    for step in range(start_step, int(cfg.train.n_steps) + 1):
        model.train()
        maybe_apply_lm_lr_schedule(cfg, step=step, optimizer=optimizer)
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        local_tokens_this_step = 0

        for _ in range(int(cfg.train.grad_accum_steps)):
            batch = benchmark.sample_train_batch(
                batch_size=int(cfg.train.batch_size),
                device=device,
                include_next_token=True,
                generator=train_generator,
            )
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="mean",
                label_smoothing=float(cfg.train.label_smoothing),
            )
            (loss / int(cfg.train.grad_accum_steps)).backward()
            loss_total += float(loss.item())
            local_tokens_this_step += int(targets.numel())

        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.train.max_grad_norm)).item())
        optimizer.step()
        reduced_train_nats, reduced_grad_norm = all_reduce_mean_scalars(
            dist_ctx,
            [loss_total / float(cfg.train.grad_accum_steps), grad_norm],
            device=device,
        )
        global_tokens_this_step = int(all_reduce_sum_scalars(dist_ctx, [float(local_tokens_this_step)], device=device)[0])
        tokens_seen += int(global_tokens_this_step)
        row = {
            "step": int(step),
            "train_nats_per_token": float(reduced_train_nats),
            "train_conservation_law_nats": None,
            "grad_norm": float(reduced_grad_norm),
            "embed_norm": lm_embedding_norm(model),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "tokens_seen": int(tokens_seen),
            "epoch_progress": float(tokens_seen / max(train_token_count, 1)),
            "val_nats_per_token": None,
            "val_conservation_law_nats": None,
            "val_evaluated_tokens": None,
            "val_evaluated_batches": None,
        }
        history_rows.append(row)

        if step == start_step or step % int(cfg.logging.log_every) == 0 or step == int(cfg.train.n_steps):
            logger.log(
                {
                    "train/ce_nats": row["train_nats_per_token"],
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

        should_eval = eval_every > 0 and step < int(cfg.train.n_steps) and step % eval_every == 0
        if should_eval:
            if eval_log_conservation_integral(cfg):
                train_eval_metrics = evaluate_causal_nats(
                    model,
                    benchmark,
                    split="train",
                    batch_size=int(cfg.eval.batch_size),
                    device=device,
                    max_batches=None if cfg.eval.max_batches is None else int(cfg.eval.max_batches),
                    dist_ctx=dist_ctx,
                )
                row["train_conservation_law_nats"] = float(
                    conservation_law_nats(objective=str(cfg.objective), metrics=train_eval_metrics)
                )
            val_metrics = evaluate_causal_nats(
                model,
                benchmark,
                split=primary_split,
                batch_size=int(cfg.eval.batch_size),
                device=device,
                max_batches=None if cfg.eval.max_batches is None else int(cfg.eval.max_batches),
                dist_ctx=dist_ctx,
            )
            row.update(
                {
                    "val_nats_per_token": float(val_metrics["nats_per_token"]),
                    "val_conservation_law_nats": float(conservation_law_nats(objective=str(cfg.objective), metrics=val_metrics)),
                    "val_evaluated_tokens": int(val_metrics["evaluated_tokens"]),
                    "val_evaluated_batches": int(val_metrics["evaluated_batches"]),
                }
            )
            logger.log(
                {
                    **(
                        {"train/conservation_law_nats": row["train_conservation_law_nats"]}
                        if row["train_conservation_law_nats"] is not None
                        else {}
                    ),
                    **(
                        {
                            "valid/nats_per_token": row["val_nats_per_token"],
                            **(
                                {"valid/conservation_law_nats": row["val_conservation_law_nats"]}
                                if eval_log_conservation_integral(cfg)
                                else {}
                            ),
                            "valid/evaluated_tokens": row["val_evaluated_tokens"],
                            "valid/evaluated_batches": row["val_evaluated_batches"],
                        }
                        if primary_split == "val"
                        else {
                            f"{primary_split}/nats_per_token": row["val_nats_per_token"],
                            **(
                                {f"{primary_split}/conservation_law_nats": row["val_conservation_law_nats"]}
                                if eval_log_conservation_integral(cfg)
                                else {}
                            ),
                            f"{primary_split}/evaluated_tokens": row["val_evaluated_tokens"],
                            f"{primary_split}/evaluated_batches": row["val_evaluated_batches"],
                        }
                    ),
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

    split_metrics: dict[str, dict[str, float]] = {}
    for split in list(cfg.eval.report_splits):
        split_name = str(split)
        split_metrics[split_name] = evaluate_causal_nats(
            model,
            benchmark,
            split=split_name,
            batch_size=int(cfg.eval.batch_size),
            device=device,
            max_batches=None if cfg.eval.max_batches is None else int(cfg.eval.max_batches),
            dist_ctx=dist_ctx,
        )
        split_metrics[split_name]["conservation_law_nats"] = conservation_law_nats(
            objective=str(cfg.objective),
            metrics=split_metrics[split_name],
        )

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
        },
        "nats_per_token": {split: float(values["nats_per_token"]) for split, values in split_metrics.items()},
        "eval_details": split_metrics,
        "ranking_metric_nats_per_token": float(split_metrics[primary_split]["nats_per_token"]),
        "primary_split": primary_split,
        **runtime_metrics(start_time=start_time, device=device),
    }
    if dist_ctx.is_main_process:
        save_csv(output_dir / cfg.paths.history_filename, history_rows)
        save_json(output_dir / cfg.paths.metrics_filename, metrics)
        torch.save(model.module.state_dict() if hasattr(model, "module") else model.state_dict(), output_dir / cfg.paths.model_filename)
        final_log_payload: dict[str, float | int | str | bool] = {}
        if "val" in split_metrics:
            final_log_payload.update(prefixed_scalar_metrics("valid", split_metrics["val"]))
        for split_name, split_values in split_metrics.items():
            if str(split_name) == "val":
                continue
            final_log_payload.update(prefixed_scalar_metrics(str(split_name), split_values))
        if final_log_payload:
            logger.log(final_log_payload, step=int(cfg.train.n_steps))
        logger.log({"eval/conservation_law_nats": float(split_metrics[primary_split]["conservation_law_nats"])}, step=int(cfg.train.n_steps))
        logger.log({"eval": metrics})
        logger.log_directory_files(output_dir, artifact_name=f"{run_id}-outputs", exclude_dirs=[str(cfg.paths.wandb_dir_name)])
    barrier(dist_ctx)
    logger.finish()
    cleanup_distributed(dist_ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
