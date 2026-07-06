"""Training callback assembly and callback implementations."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import torch
from omegaconf import DictConfig

from src.callbacks import MetricHistory

from .experiment_registry import require_experiment_kind
from .evaluation import evaluate_model, save_mmse_curve_plot


TrainingCallback = Callable[[int, torch.nn.Module, MetricHistory, torch.optim.Optimizer], bool]


def build_training_callback(
    cfg: DictConfig,
    device: torch.device,
    embed_dim: int,
    data_spec,
    output_dir: Path,
    logger,
    verbose: bool,
):
    require_experiment_kind(cfg, "immse")
    callbacks: list[TrainingCallback] = []
    checkpoint_callback = _build_checkpoint_callback(
        cfg=cfg,
        output_dir=output_dir,
        logger=logger,
    )
    if checkpoint_callback is not None:
        callbacks.append(checkpoint_callback)

    min_distance_callback = _build_print_min_embed_distance_callback(cfg, logger=logger)
    if min_distance_callback is not None:
        callbacks.append(min_distance_callback)

    plot_callback = _build_plot_immse_curve_callback(
        cfg=cfg,
        device=device,
        embed_dim=embed_dim,
        data_spec=data_spec,
        output_dir=output_dir,
        logger=logger,
        verbose=verbose,
    )
    if plot_callback is not None:
        callbacks.append(plot_callback)

    early_stop_callback = _build_early_stop_callback(
        cfg=cfg,
        device=device,
        embed_dim=embed_dim,
        data_spec=data_spec,
        logger=logger,
        verbose=verbose,
    )
    if early_stop_callback is not None:
        callbacks.append(early_stop_callback)

    lr_decay_callback = _build_lr_decay_callback(
        cfg=cfg,
        device=device,
        embed_dim=embed_dim,
        data_spec=data_spec,
        logger=logger,
        verbose=verbose,
    )
    if lr_decay_callback is not None:
        callbacks.append(lr_decay_callback)

    if not callbacks:
        return None

    def _callback(
        step: int,
        current_model: torch.nn.Module,
        history: MetricHistory,
        optimizer: torch.optim.Optimizer,
    ) -> bool:
        should_stop = False
        for callback in callbacks:
            should_stop = callback(step, current_model, history, optimizer) or should_stop
        return should_stop

    return _callback


def _build_checkpoint_callback(
    cfg: DictConfig,
    output_dir: Path,
    logger,
):
    save_every = getattr(cfg.train, "save_every", None)
    if save_every is None:
        return None

    save_every = max(1, int(save_every))
    checkpoint_dir = output_dir / str(cfg.paths.checkpoint_dir_name)
    filename_template = str(cfg.paths.checkpoint_filename_template)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _checkpoint_callback(
        step: int,
        current_model: torch.nn.Module,
        history: MetricHistory,
        optimizer: torch.optim.Optimizer,
    ) -> bool:
        del history
        del optimizer
        if step % save_every != 0:
            return False

        output_path = checkpoint_dir / filename_template.format(step=int(step))
        torch.save(current_model.state_dict(), output_path)
        if logger is not None:
            logger.log({"monitor/checkpoint_step": int(step)}, step=step)
        print(f"[checkpoint] step={step} saved={output_path}")
        return False

    return _checkpoint_callback


def _build_early_stop_callback(
    cfg: DictConfig,
    device: torch.device,
    embed_dim: int,
    data_spec,
    logger,
    verbose: bool,
):
    require_experiment_kind(cfg, "immse")
    if not bool(cfg.callbacks.early_stopping.enabled):
        return None

    min_steps = int(cfg.callbacks.early_stopping.min_steps)
    check_every = max(1, int(cfg.callbacks.early_stopping.check_every))
    rel_error_threshold = float(cfg.callbacks.early_stopping.rel_error_threshold)
    early_stop_eval_cfg = _train_eval_overrides(cfg, cfg.callbacks.early_stopping)

    def _stop_callback(
        step: int,
        current_model: torch.nn.Module,
        history: MetricHistory,
        optimizer: torch.optim.Optimizer,
    ) -> bool:
        del optimizer
        del history
        if step < min_steps or step % check_every != 0:
            return False

        _, _, eval_info = evaluate_model(
            cfg,
            current_model,
            device,
            embed_dim,
            data_spec,
            eval_overrides=early_stop_eval_cfg,
        )
        model_area = float(eval_info["model_area_nats"])
        reference_entropy = float(eval_info["reference_entropy_nats"])
        rel_error = abs(model_area - reference_entropy) / max(abs(reference_entropy), 1e-12)
        if logger is not None:
            logger.log(
                {
                    "monitor/early_stop_model_area_nats": model_area,
                    "monitor/early_stop_reference_entropy_nats": reference_entropy,
                    "monitor/early_stop_rel_error": rel_error,
                },
                step=step,
            )
        print(
            f"[early-stop-check] step={step} model_area={model_area:.6f} "
            f"reference_entropy={reference_entropy:.6f} "
            f"rel_error={rel_error:.2%}"
        )
        if rel_error <= rel_error_threshold:
            print(
                f"[early-stop] step={step} rel_error={rel_error:.2%} "
                f"<= threshold={rel_error_threshold:.2%}"
            )
            return True
        return False

    return _stop_callback


def _build_print_min_embed_distance_callback(cfg: DictConfig, logger):
    require_experiment_kind(cfg, "immse")
    if not bool(cfg.callbacks.print_min_embed_distance.enabled):
        return None

    check_every = max(1, int(cfg.callbacks.print_min_embed_distance.check_every))
    large_alphabet_threshold = int(cfg.callbacks.print_min_embed_distance.large_alphabet_threshold)
    batch_size = int(cfg.train.batch_size)

    def _print_callback(
        step: int,
        current_model: torch.nn.Module,
        history: MetricHistory,
        optimizer: torch.optim.Optimizer,
    ) -> bool:
        del optimizer
        del history
        if step % check_every != 0:
            return False

        min_distance = _min_embedding_distance(
            current_model.x_values.weight,
            large_alphabet_threshold=large_alphabet_threshold,
            batch_size=batch_size,
        )
        if logger is not None:
            logger.log({"monitor/min_embed_distance": min_distance}, step=step)
        print(f"[min-embed-distance] step={step} min_distance={min_distance:.6f}")
        return False

    return _print_callback


def _build_plot_immse_curve_callback(
    cfg: DictConfig,
    device: torch.device,
    embed_dim: int,
    data_spec,
    output_dir: Path,
    logger,
    verbose: bool,
):
    require_experiment_kind(cfg, "immse")
    if not bool(cfg.callbacks.plot_immse_curve.enabled):
        return None

    check_every = max(1, int(cfg.callbacks.plot_immse_curve.check_every))
    plot_dir = output_dir / str(cfg.callbacks.plot_immse_curve.dir_name)
    filename_template = str(cfg.callbacks.plot_immse_curve.filename_template)
    plot_eval_cfg = _train_eval_overrides(cfg, cfg.callbacks.plot_immse_curve)
    plot_dir.mkdir(parents=True, exist_ok=True)

    def _plot_callback(
        step: int,
        current_model: torch.nn.Module,
        history: MetricHistory,
        optimizer: torch.optim.Optimizer,
    ) -> bool:
        del optimizer
        del history
        if step % check_every != 0:
            return False

        snr_grid, curves, eval_info = evaluate_model(
            cfg,
            current_model,
            device,
            embed_dim,
            data_spec,
            eval_overrides=plot_eval_cfg,
        )
        reference_area = float(eval_info["reference_area_nats"])
        model_area = float(eval_info["model_area_nats"])
        output_path = plot_dir / filename_template.format(step=int(step))
        save_mmse_curve_plot(
            output_path=output_path,
            snr_grid=snr_grid,
            true_mmse=curves["true_mmse"],
            model_mmse=curves["model_mmse"],
            true_area=reference_area,
            model_area=model_area,
            title=eval_info["title"],
        )
        if logger is not None:
            logger.log(
                {
                    "monitor/reference_area_nats": reference_area,
                    "monitor/model_area_nats": model_area,
                    "monitor/model_rel_error": eval_info["model_rel_error"],
                    "monitor/reference_entropy_nats": eval_info["reference_entropy_nats"],
                },
                step=step,
            )
            logger.log_image("monitor/mmse_curve", output_path, step=step, caption=f"step={step}")
        print(f"[plot-immse-curve] step={step} saved={output_path}")
        return False

    return _plot_callback


def _train_eval_overrides(cfg: DictConfig, callback_cfg: DictConfig) -> SimpleNamespace:
    def _callback_value(name: str) -> int:
        value = getattr(callback_cfg, name, None)
        if value is None:
            return int(getattr(cfg.eval, name))
        return int(value)

    require_experiment_kind(cfg, "immse")
    train_seq_len = getattr(cfg.train, "seq_len", getattr(cfg.eval, "seq_len", None))
    if train_seq_len is None:
        raise ValueError("Expected train.seq_len or eval.seq_len to be set for callback evaluation.")
    return SimpleNamespace(
        n_snr_points=_callback_value("n_snr_points"),
        n_samples=_callback_value("n_samples"),
        batch_size=_callback_value("batch_size"),
        seq_len=int(train_seq_len),
    )


def _build_lr_decay_callback(
    cfg: DictConfig,
    device: torch.device,
    embed_dim: int,
    data_spec,
    logger,
    verbose: bool,
):
    require_experiment_kind(cfg, "immse")
    if not bool(cfg.callbacks.lr_decay.enabled):
        return None

    check_every = max(1, int(cfg.callbacks.lr_decay.check_every))
    warmup_steps = max(0, int(cfg.callbacks.lr_decay.warmup_steps))
    min_lr = float(cfg.callbacks.lr_decay.min_lr)
    total_steps = max(1, int(cfg.train.n_steps))
    del device, embed_dim, data_spec

    def _lr_decay_callback(
        step: int,
        current_model: torch.nn.Module,
        history: MetricHistory,
        optimizer: torch.optim.Optimizer,
    ) -> bool:
        del current_model
        del history
        if step <= 0:
            return False

        base_lr = float(optimizer.param_groups[0].setdefault("initial_lr", optimizer.param_groups[0]["lr"]))
        if warmup_steps > 0 and step <= warmup_steps:
            new_lr = base_lr * (float(step) / float(warmup_steps))
        else:
            cosine_steps = max(1, total_steps - warmup_steps)
            cosine_progress = min(1.0, max(0.0, float(step - warmup_steps) / float(cosine_steps)))
            cosine = 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
            new_lr = min_lr + (base_lr - min_lr) * cosine
        new_lr = max(min_lr, new_lr)

        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr

        should_log = logger is not None and (step == 1 or step % check_every == 0 or step == total_steps)
        if should_log:
            logger.log({"monitor/lr": new_lr}, step=step)
        if step == 1 or step % check_every == 0 or step == total_steps:
            print(f"[lr-decay] step={step} lr={new_lr:.6g}")
        return False

    return _lr_decay_callback


def _min_embedding_distance(weight: torch.Tensor, large_alphabet_threshold: int, batch_size: int) -> float:
    weight = weight.detach()
    n_embeddings = int(weight.shape[0])
    if n_embeddings < 2:
        return float("inf")

    if n_embeddings <= int(large_alphabet_threshold):
        distances = torch.cdist(weight, weight)
        distances.fill_diagonal_(float("inf"))
        return float(distances.min().item())

    block_size = max(1, int(batch_size))
    norms = torch.sum(weight * weight, dim=1)
    min_distance_sq = float("inf")
    for start_i in range(0, n_embeddings, block_size):
        end_i = min(start_i + block_size, n_embeddings)
        block_i = weight[start_i:end_i]
        norms_i = norms[start_i:end_i].unsqueeze(1)
        for start_j in range(start_i, n_embeddings, block_size):
            end_j = min(start_j + block_size, n_embeddings)
            block_j = weight[start_j:end_j]
            norms_j = norms[start_j:end_j].unsqueeze(0)
            distances_sq = torch.clamp(norms_i + norms_j - 2.0 * (block_i @ block_j.T), min=0.0)
            if start_i == start_j:
                diag = torch.arange(end_i - start_i, device=weight.device)
                distances_sq[diag, diag] = float("inf")
            block_min_sq = float(distances_sq.min().item())
            if block_min_sq < min_distance_sq:
                min_distance_sq = block_min_sq
    return math.sqrt(min_distance_sq)
