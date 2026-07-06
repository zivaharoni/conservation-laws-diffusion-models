from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Callable
import math

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "gexit-diffusion-matplotlib"))

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common.config import compose_config, resolve_config_name_and_overrides
from experiments.common.experiment_registry import require_experiment_kind
from experiments.common.io import deterministic_output_dir, save_config, save_csv, save_json
from experiments.common.repro import configure_environment, get_best_device, make_run_id, set_global_seed
from experiments.common.tikz_export import export_curve_table
from experiments.common.wandb_logger import WandbLogger
from experiments.common.training_callbacks import _min_embedding_distance
from src.exit import (
    apply_sequence_masking,
    curve_area_metrics,
    evaluate_model_exit,
    evaluate_model_exit_curve,
    evaluate_reference_iid_exit_curve,
    evaluate_reference_markov_exit_curve,
    per_token_nll,
    sample_uniform_mask_rates,
    save_exit_curve_plot,
)
from src.models import BidirectionalMaskedTokenModel
from src.sources_iid import entropy_nats, sample_dirichlet_iid_probs, uniform_iid_probs
from src.sources_markov import build_markov_chain, sample_markov_sequences_torch


class FamilySampler:
    def __init__(
        self,
        *,
        family: str,
        metadata: dict[str, object],
        reference_entropy_nats: float,
        train_sampler: Callable[[int, int, torch.device | str], torch.Tensor],
        eval_sampler: Callable[[int, int, torch.device | str], torch.Tensor],
        source_probs: torch.Tensor | None = None,
        init_probs: torch.Tensor | None = None,
        stationary_dist: torch.Tensor | None = None,
        transition_matrix: torch.Tensor | None = None,
        sequence_permutation: torch.Tensor | None = None,
    ) -> None:
        self.family = family
        self.metadata = metadata
        self.reference_entropy_nats = reference_entropy_nats
        self._train_sampler = train_sampler
        self._eval_sampler = eval_sampler
        self.source_probs = source_probs
        self.init_probs = init_probs
        self.stationary_dist = stationary_dist
        self.transition_matrix = transition_matrix
        self.sequence_permutation = sequence_permutation

    def sample_train(self, batch_size: int, seq_len: int, device: torch.device | str) -> torch.Tensor:
        return self._train_sampler(batch_size, seq_len, device)

    def sample_eval(self, batch_size: int, seq_len: int, device: torch.device | str) -> torch.Tensor:
        return self._eval_sampler(batch_size, seq_len, device)


def _count_parameters(model: nn.Module) -> int:
    return sum(int(param.numel()) for param in model.parameters())


def _format_tensor_preview(tensor: torch.Tensor, max_items: int = 6) -> str:
    flat = torch.as_tensor(tensor, dtype=torch.float64).detach().cpu().reshape(-1)
    if flat.numel() <= max_items:
        return "[" + ", ".join(f"{float(value):.4f}" for value in flat.tolist()) + "]"
    head_count = max(1, max_items // 2)
    tail_count = max(1, max_items - head_count)
    head = ", ".join(f"{float(value):.4f}" for value in flat[:head_count].tolist())
    tail = ", ".join(f"{float(value):.4f}" for value in flat[-tail_count:].tolist())
    return f"[{head}, ..., {tail}]"


def _print_reference_summary(sampler: FamilySampler) -> None:
    rows: list[tuple[str, str]] = [
        ("Alphabet Size", str(int(sampler.metadata["alphabet_size"]))),
    ]
    if sampler.family == "iid":
        rows.append(("Source Distribution", str(sampler.metadata["source_distribution"])))
        dirichlet_alpha = sampler.metadata.get("dirichlet_alpha")
        if dirichlet_alpha is not None:
            rows.append(("Dirichlet Alpha", f"{float(dirichlet_alpha):.4f}"))
        if sampler.source_probs is not None:
            rows.append(("Source Probs", _format_tensor_preview(sampler.source_probs)))
        reference_unit = "nats"
    else:
        rows.append(("Dirichlet Alpha", f"{float(sampler.metadata['dirichlet_alpha']):.4f}"))
        rows.append(("Init Distribution", str(sampler.metadata["init_distribution"])))
        if sampler.init_probs is not None:
            rows.append(("Init Probs", _format_tensor_preview(sampler.init_probs)))
        if sampler.stationary_dist is not None:
            rows.append(("Stationary Dist", _format_tensor_preview(sampler.stationary_dist)))
        if sampler.transition_matrix is not None:
            rows.append(("Transition Row0", _format_tensor_preview(sampler.transition_matrix[0])))
        reference_unit = "nats/symbol"

    for label, value in rows:
        print(f"{label + ':':<24}{value}")
    print(f"{'Reference entropy:':<24}{float(sampler.reference_entropy_nats):.6f} {reference_unit}")
    if sampler.stationary_dist is not None:
        stationary_entropy_nats = entropy_nats(sampler.stationary_dist)
        print(f"{'Stationary entropy:':<24}{stationary_entropy_nats:.6f} nats")


def _sample_sequence_permutation(seq_len: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randperm(int(seq_len), generator=generator, dtype=torch.long)


def _polar_bit_reversal_permutation(seq_len: int) -> torch.Tensor:
    seq_len = int(seq_len)
    if seq_len <= 0 or (seq_len & (seq_len - 1)) != 0:
        raise ValueError(f"Polar bit-reversal requires a power-of-two seq_len, got {seq_len}")
    n_bits = seq_len.bit_length() - 1
    idx = torch.arange(seq_len, dtype=torch.long)
    permutation = torch.zeros_like(idx)
    for bit in range(n_bits):
        permutation |= ((idx >> bit) & 1) << (n_bits - 1 - bit)
    return permutation


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


def _model_kwargs_from_cfg(cfg: DictConfig) -> dict[str, object]:
    max_seq_len = cfg.model.max_seq_len
    if max_seq_len is None:
        max_seq_len = cfg.train.seq_len
    return {
        "alphabet_size": int(cfg.data.alphabet_size),
        "embed_dim": int(cfg.model.embed_dim),
        "n_transformer_layers": int(cfg.model.n_transformer_layers),
        "n_heads": int(cfg.model.n_heads),
        "attention_dropout": float(cfg.model.attention_dropout),
        "max_seq_len": int(max_seq_len),
    }


def _build_model(cfg: DictConfig, *, project_root: Path, verbose: bool = False) -> BidirectionalMaskedTokenModel:
    model = BidirectionalMaskedTokenModel(**_model_kwargs_from_cfg(cfg))
    init_weights_path = cfg.train.get("init_weights_path")
    if init_weights_path not in (None, ""):
        checkpoint_path = _resolve_init_weights_path(str(init_weights_path), project_root=project_root)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Initial weights file not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = _extract_state_dict(checkpoint)
        strict = bool(cfg.train.get("init_weights_strict", True))
        model.load_state_dict(state_dict, strict=strict)
        if verbose:
            print(f"[setup] initialized model weights from {checkpoint_path}")
    return model


def _validate_config(cfg: DictConfig) -> None:
    require_experiment_kind(cfg, "exit")
    family = str(cfg.family)
    if family not in {"iid", "markov", "permuted_markov", "polar_markov"}:
        raise ValueError(f"Unsupported exit.family={family!r}.")
    if int(cfg.data.alphabet_size) <= 0:
        raise ValueError(f"data.alphabet_size must be positive, got {cfg.data.alphabet_size}.")
    if int(cfg.train.seq_len) <= 0:
        raise ValueError(f"train.seq_len must be positive, got {cfg.train.seq_len}.")
    if int(cfg.train.n_steps) <= 0:
        raise ValueError(f"train.n_steps must be positive, got {cfg.train.n_steps}.")
    if int(cfg.train.batch_size) <= 0:
        raise ValueError(f"train.batch_size must be positive, got {cfg.train.batch_size}.")
    if float(cfg.train.lr) <= 0.0:
        raise ValueError(f"train.lr must be positive, got {cfg.train.lr}.")
    if int(cfg.eval.eval_batches) <= 0:
        raise ValueError(f"eval.eval_batches must be positive, got {cfg.eval.eval_batches}.")
    if int(cfg.eval.eval_batch_size) <= 0:
        raise ValueError(
            f"eval.eval_batch_size must be positive, got {cfg.eval.eval_batch_size}."
        )
    if int(cfg.eval.n_s_points) < 2:
        raise ValueError(f"eval.n_s_points must be at least 2, got {cfg.eval.n_s_points}.")

    grad_clip_cfg = cfg.train.grad_clip
    grad_clip_values = [
        float(grad_clip_cfg.embedding),
        float(grad_clip_cfg.attention),
        float(grad_clip_cfg.alphabet_proj),
        float(grad_clip_cfg.other),
    ]
    if any(value <= 0.0 for value in grad_clip_values):
        raise ValueError(f"train.grad_clip values must be positive, got {grad_clip_values}.")

    if family == "polar_markov":
        seq_len = int(cfg.train.seq_len)
        if seq_len <= 0 or (seq_len & (seq_len - 1)) != 0:
            raise ValueError(f"train.seq_len must be a power of two for polar_markov, got {seq_len}.")

    if family == "iid":
        source_distribution = str(cfg.data.source_distribution)
        if source_distribution not in {"uniform", "dirichlet"}:
            raise ValueError(
                f"data.source_distribution must be 'uniform' or 'dirichlet' for iid EXIT, got {source_distribution!r}."
            )
    else:
        if "dirichlet_alpha" not in cfg.data:
            raise ValueError(f"Markov EXIT config requires data.dirichlet_alpha, got {OmegaConf.to_yaml(cfg.data)}")
        if "init_distribution" not in cfg.data:
            raise ValueError(f"Markov EXIT config requires data.init_distribution, got {OmegaConf.to_yaml(cfg.data)}")


def _make_iid_sampler(cfg: DictConfig) -> FamilySampler:
    alphabet_size = int(cfg.data.alphabet_size)
    source_distribution = str(cfg.data.source_distribution)
    seed = int(cfg.seed)
    dirichlet_alpha = cfg.data.get("dirichlet_alpha")

    if source_distribution == "uniform":
        probs = uniform_iid_probs(alphabet_size)
    elif source_distribution == "dirichlet":
        if dirichlet_alpha is None:
            raise ValueError("data.dirichlet_alpha must be set for dirichlet iid EXIT.")
        probs = sample_dirichlet_iid_probs(alphabet_size, float(dirichlet_alpha), seed)
    else:
        raise ValueError(f"Unsupported data.source_distribution={source_distribution!r}.")

    probs_cpu = torch.as_tensor(probs, dtype=torch.float32, device="cpu")
    train_generator = torch.Generator(device="cpu")
    train_generator.manual_seed(seed + 101)
    eval_generator = torch.Generator(device="cpu")
    eval_generator.manual_seed(seed + 102)

    def _sample_with_generator(
        generator: torch.Generator,
        batch_size: int,
        seq_len: int,
        device: torch.device | str,
    ) -> torch.Tensor:
        states = torch.multinomial(
            probs_cpu,
            num_samples=int(batch_size * seq_len),
            replacement=True,
            generator=generator,
        ).view(batch_size, seq_len)
        return states.to(device=device, dtype=torch.long)

    metadata = {
        "family": "iid",
        "source_distribution": source_distribution,
        "alphabet_size": alphabet_size,
        "dirichlet_alpha": None if dirichlet_alpha is None else float(dirichlet_alpha),
    }
    return FamilySampler(
        family="iid",
        metadata=metadata,
        reference_entropy_nats=float(entropy_nats(probs)),
        train_sampler=lambda batch_size, seq_len, device: _sample_with_generator(train_generator, batch_size, seq_len, device),
        eval_sampler=lambda batch_size, seq_len, device: _sample_with_generator(eval_generator, batch_size, seq_len, device),
        source_probs=torch.as_tensor(probs, dtype=torch.float64, device="cpu"),
    )


def _make_markov_sampler(cfg: DictConfig) -> FamilySampler:
    family = str(cfg.family)
    alphabet_size = int(cfg.data.alphabet_size)
    seq_len = int(cfg.train.seq_len)
    dirichlet_alpha = float(cfg.data.dirichlet_alpha)
    seed = int(cfg.seed)
    init_distribution = str(cfg.data.init_distribution)

    chain = build_markov_chain(
        alphabet_size=alphabet_size,
        dirichlet_alpha=dirichlet_alpha,
        seed=seed,
        init_distribution=init_distribution,
    )
    if family == "markov":
        permutation = None
    elif family == "permuted_markov":
        permutation = _sample_sequence_permutation(seq_len, seed=seed + 10_000)
    elif family == "polar_markov":
        permutation = _polar_bit_reversal_permutation(seq_len)
    else:
        raise ValueError(f"Unsupported exit.family={family!r}.")

    init_probs = torch.as_tensor(chain.init_probs, dtype=torch.float32, device="cpu")
    transition_matrix = torch.as_tensor(chain.transition_matrix, dtype=torch.float32, device="cpu")
    permutation_cpu = None if permutation is None else torch.as_tensor(permutation, dtype=torch.long, device="cpu")

    train_generator = torch.Generator(device="cpu")
    train_generator.manual_seed(seed + 201)
    eval_generator = torch.Generator(device="cpu")
    eval_generator.manual_seed(seed + 202)

    def _sample_with_generator(
        generator: torch.Generator,
        batch_size: int,
        seq_len: int,
        device: torch.device | str,
    ) -> torch.Tensor:
        uniforms = torch.rand((batch_size, seq_len), generator=generator, dtype=torch.float32)
        states = sample_markov_sequences_torch(
            num_sequences=batch_size,
            seq_len=seq_len,
            init_probs=init_probs,
            transition_matrix=transition_matrix,
            uniforms=uniforms,
        )
        if permutation_cpu is not None:
            states = states[:, permutation_cpu]
        return states.to(device=device, dtype=torch.long)

    metadata = {
        "family": family,
        "alphabet_size": alphabet_size,
        "dirichlet_alpha": dirichlet_alpha,
        "init_distribution": init_distribution,
    }
    if permutation_cpu is not None:
        metadata["sequence_permutation_preview"] = [int(value) for value in permutation_cpu[: min(8, seq_len)].tolist()]
        if family == "polar_markov":
            metadata["sequence_transform"] = "polar_bit_reversal"

    return FamilySampler(
        family=family,
        metadata=metadata,
        reference_entropy_nats=float(chain.entropy_rate_nats),
        train_sampler=lambda batch_size, seq_len, device: _sample_with_generator(train_generator, batch_size, seq_len, device),
        eval_sampler=lambda batch_size, seq_len, device: _sample_with_generator(eval_generator, batch_size, seq_len, device),
        init_probs=init_probs.to(dtype=torch.float64),
        stationary_dist=torch.as_tensor(chain.stationary_dist, dtype=torch.float64, device="cpu"),
        transition_matrix=transition_matrix.to(dtype=torch.float64),
        sequence_permutation=None if permutation_cpu is None else permutation_cpu.clone(),
    )


def _build_family_sampler(cfg: DictConfig) -> FamilySampler:
    require_experiment_kind(cfg, "exit")
    family = str(cfg.family)
    if family == "iid":
        return _make_iid_sampler(cfg)
    return _make_markov_sampler(cfg)


def _resolve_grad_clip(cfg: DictConfig) -> float:
    grad_clip_cfg = cfg.train.grad_clip
    return max(
        float(grad_clip_cfg.embedding),
        float(grad_clip_cfg.attention),
        float(grad_clip_cfg.alphabet_proj),
        float(grad_clip_cfg.other),
    )


def _embedding_norm(model: nn.Module) -> float:
    if not hasattr(model, "x_values"):
        return float("nan")
    return float(model.x_values.weight.detach().norm(dim=-1).mean().item())


def _run_exit_evaluation(
    cfg: DictConfig,
    *,
    model: BidirectionalMaskedTokenModel,
    sampler: FamilySampler,
    device: torch.device,
    seed_offset: int = 10_000,
) -> tuple[dict[str, float], torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, float], list[dict[str, float]]]:
    eval_metrics = evaluate_model_exit(
        model,
        sampler.sample_eval,
        batch_size=int(cfg.eval.eval_batch_size),
        seq_len=int(cfg.train.seq_len),
        eval_batches=int(cfg.eval.eval_batches),
        device=device,
        mask_token_id=model.mask_token_id,
        seed=int(cfg.seed) + seed_offset,
        fixed_s=None,
        steps_per_batch=int(getattr(cfg.eval, "steps_per_batch", 1)),
    )
    s_grid = torch.linspace(0.0, 1.0, steps=int(cfg.eval.n_s_points), dtype=torch.float64)
    model_curve = evaluate_model_exit_curve(
        model,
        sampler.sample_eval,
        s_grid=s_grid,
        batch_size=int(cfg.eval.eval_batch_size),
        seq_len=int(cfg.train.seq_len),
        eval_batches=int(cfg.eval.eval_batches),
        device=device,
        mask_token_id=model.mask_token_id,
        seed=int(cfg.seed) + seed_offset + 10_000,
        steps_per_batch=int(getattr(cfg.eval, "steps_per_batch", 1)),
    )
    if sampler.family == "iid":
        reference_curve = evaluate_reference_iid_exit_curve(sampler.source_probs, s_grid=s_grid)
    else:
        reference_curve = evaluate_reference_markov_exit_curve(
            sampler.sample_eval,
            init_probs=sampler.init_probs,
            transition_matrix=sampler.transition_matrix,
            s_grid=s_grid,
            batch_size=int(cfg.eval.eval_batch_size),
            seq_len=int(cfg.train.seq_len),
            eval_batches=int(cfg.eval.eval_batches),
            seed=int(cfg.seed) + seed_offset + 20_000,
            sequence_permutation=sampler.sequence_permutation,
        )

    curve_metrics = curve_area_metrics(
        s_grid=s_grid,
        reference_exit=reference_curve["exit_gap"],
        model_exit=model_curve["exit_gap"],
        reference_entropy=float(sampler.reference_entropy_nats),
    )
    curve_rows = [
        {
            "s": float(s),
            "true_masked_ce": float(true_masked),
            "true_unmasked_ce": float(true_unmasked),
            "true_exit": float(true_exit),
            "model_masked_ce": float(model_masked),
            "model_unmasked_ce": float(model_unmasked),
            "model_exit": float(model_exit),
        }
        for s, true_masked, true_unmasked, true_exit, model_masked, model_unmasked, model_exit in zip(
            s_grid.tolist(),
            reference_curve["masked_ce"].tolist(),
            reference_curve["unmasked_ce"].tolist(),
            reference_curve["exit_gap"].tolist(),
            model_curve["masked_ce"].tolist(),
            model_curve["unmasked_ce"].tolist(),
            model_curve["exit_gap"].tolist(),
        )
    ]
    return eval_metrics, s_grid, model_curve, reference_curve, curve_metrics, curve_rows


def _maybe_apply_lr_decay(cfg: DictConfig, *, step: int, optimizer: torch.optim.Optimizer, logger: WandbLogger | None) -> None:
    require_experiment_kind(cfg, "exit")
    if not bool(cfg.callbacks.lr_decay.enabled):
        return

    warmup_steps = max(0, int(cfg.callbacks.lr_decay.warmup_steps))
    min_lr = float(cfg.callbacks.lr_decay.min_lr)
    total_steps = max(1, int(cfg.train.n_steps))
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

    print_every = max(1, int(getattr(cfg.logging, "print_every", 1)))
    if step == 1 or step % print_every == 0 or step == total_steps:
        if logger is not None:
            logger.log({"monitor/lr": new_lr}, step=step)
        print(f"[lr-decay] step={step} lr={new_lr:.6g}")


def _maybe_log_min_embed_distance(
    cfg: DictConfig,
    *,
    step: int,
    model: BidirectionalMaskedTokenModel,
    logger: WandbLogger | None,
) -> None:
    require_experiment_kind(cfg, "exit")
    if not bool(cfg.callbacks.print_min_embed_distance.enabled):
        return
    check_every = max(1, int(cfg.callbacks.print_min_embed_distance.check_every))
    if step % check_every != 0:
        return
    min_distance = _min_embedding_distance(
        model.x_values.weight,
        large_alphabet_threshold=int(cfg.callbacks.print_min_embed_distance.large_alphabet_threshold),
        batch_size=int(cfg.train.batch_size),
    )
    if logger is not None:
        logger.log({"monitor/min_embed_distance": min_distance}, step=step)
    print(f"[min-embed-distance] step={step} min_distance={min_distance:.6f}")


def _maybe_plot_exit_curve(
    cfg: DictConfig,
    *,
    step: int,
    model: BidirectionalMaskedTokenModel,
    sampler: FamilySampler,
    device: torch.device,
    output_dir: Path,
    logger: WandbLogger | None,
) -> None:
    require_experiment_kind(cfg, "exit")
    if not bool(cfg.callbacks.plot_exit_curve.enabled):
        return
    check_every = max(1, int(cfg.callbacks.plot_exit_curve.check_every))
    if step % check_every != 0:
        return
    _, s_grid, model_curve, reference_curve, curve_metrics, _ = _run_exit_evaluation(
        cfg,
        model=model,
        sampler=sampler,
        device=device,
        seed_offset=40_000 + int(step),
    )
    plot_dir = output_dir / str(cfg.callbacks.plot_exit_curve.dir_name)
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_path = plot_dir / str(cfg.callbacks.plot_exit_curve.filename_template).format(step=int(step))
    save_exit_curve_plot(
        output_path,
        s_grid=s_grid,
        true_exit=reference_curve["exit_gap"],
        model_exit=model_curve["exit_gap"],
        true_area=curve_metrics["reference_exit_area_nats"],
        model_area=curve_metrics["eval_exit_area_nats"],
        entropy=float(sampler.reference_entropy_nats),
        title=f"EXIT Curves ({sampler.family})",
    )
    if logger is not None:
        logger.log(
            {
                "monitor/reference_entropy_nats": curve_metrics["reference_entropy_nats"],
                "monitor/reference_exit_area_nats": curve_metrics["reference_exit_area_nats"],
                "monitor/eval_exit_area_nats": curve_metrics["eval_exit_area_nats"],
                "monitor/model_rel_error": curve_metrics["model_rel_error"],
            },
            step=step,
        )
        logger.log_image("monitor/exit_curve", output_path, step=step, caption=f"step={step}")
    print(f"[plot-exit-curve] step={step} saved={output_path}")


def _maybe_early_stop(
    cfg: DictConfig,
    *,
    step: int,
    model: BidirectionalMaskedTokenModel,
    sampler: FamilySampler,
    device: torch.device,
    logger: WandbLogger | None,
) -> bool:
    require_experiment_kind(cfg, "exit")
    if not bool(cfg.callbacks.early_stopping.enabled):
        return False
    min_steps = int(cfg.callbacks.early_stopping.min_steps)
    check_every = max(1, int(cfg.callbacks.early_stopping.check_every))
    if step < min_steps or step % check_every != 0:
        return False
    _, _, _, _, curve_metrics, _ = _run_exit_evaluation(
        cfg,
        model=model,
        sampler=sampler,
        device=device,
        seed_offset=80_000 + int(step),
    )
    model_area = float(curve_metrics["eval_exit_area_nats"])
    reference_entropy = float(curve_metrics["reference_entropy_nats"])
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
        f"reference_entropy={reference_entropy:.6f} rel_error={rel_error:.2%}"
    )
    if rel_error <= float(cfg.callbacks.early_stopping.rel_error_threshold):
        print(
            f"[early-stop] step={step} rel_error={rel_error:.2%} "
            f"<= threshold={float(cfg.callbacks.early_stopping.rel_error_threshold):.2%}"
        )
        return True
    return False


def run_experiment(
    cfg: DictConfig,
    *,
    config_name: str,
    overrides: list[str] | None = None,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, object]:
    require_experiment_kind(cfg, "exit")
    configure_environment()
    overrides = [] if overrides is None else list(overrides)
    cfg.paths.results_root = str((Path(project_root) / str(cfg.paths.results_root)).resolve())
    _validate_config(cfg)
    set_global_seed(cfg.seed, deterministic=cfg.deterministic)
    device = get_best_device(cfg.device)
    sampler = _build_family_sampler(cfg)
    run_id = make_run_id(config_name, overrides)
    output_dir = deterministic_output_dir(project_root, cfg.paths.results_root, cfg.name, run_id)

    if cfg.verbose:
        print(f"[setup] family={sampler.family}")
        print(f"[setup] run_id={run_id}")
        print(f"[setup] device={device}")
        print(f"[setup] output_dir={output_dir}")
        print("[setup] resolved config:")
        print(OmegaConf.to_yaml(cfg, resolve=True).strip())

    save_config(output_dir / cfg.paths.config_snapshot_name, cfg)

    logger = WandbLogger(cfg=cfg, run_id=run_id, output_dir=str(output_dir / cfg.paths.wandb_dir_name))
    model = _build_model(cfg, project_root=project_root, verbose=bool(cfg.verbose)).to(device)
    model_params = _count_parameters(model)
    logger.log({"model/parameter_count": model_params})
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.train.lr))
    grad_clip = _resolve_grad_clip(cfg)
    history_rows: list[dict[str, float | int]] = []
    mask_generator = torch.Generator(device="cpu")
    mask_generator.manual_seed(int(cfg.seed) + 303)

    _print_reference_summary(sampler)

    for step in range(1, int(cfg.train.n_steps) + 1):
        model.train()
        _maybe_apply_lr_decay(cfg, step=step, optimizer=optimizer, logger=logger)
        optimizer.zero_grad(set_to_none=True)

        tokens = sampler.sample_train(
            batch_size=int(cfg.train.batch_size),
            seq_len=int(cfg.train.seq_len),
            device=device,
        )
        s_per_sequence = sample_uniform_mask_rates(int(cfg.train.batch_size), device=device, generator=mask_generator)
        masked_tokens, mask, forced_count = apply_sequence_masking(
            tokens,
            s_per_sequence,
            mask_token_id=model.mask_token_id,
            generator=mask_generator,
        )
        logits = model(masked_tokens)
        losses = per_token_nll(logits, tokens)
        mask_f = mask.to(dtype=losses.dtype)
        unmasked_f = (~mask).to(dtype=losses.dtype)
        masked_count = mask_f.sum(dim=1, keepdim=True)
        unmasked_count = unmasked_f.sum(dim=1, keepdim=True)
        masked_loss_sum = torch.sum(losses * mask_f, dim=1, keepdim=True)
        unmasked_loss_sum = torch.sum(losses * unmasked_f, dim=1, keepdim=True)
        masked_ce_tensor = masked_loss_sum / torch.clamp(masked_count, min=1.0)
        unmasked_ce_tensor = unmasked_loss_sum / torch.clamp(unmasked_count, min=1.0)
        loss = (masked_ce_tensor + unmasked_ce_tensor).mean()
        loss.backward()

        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip).item())
        optimizer.step()

        masked_ce = float("nan") if masked_count.sum() <= 0 else float(masked_ce_tensor.mean().item())
        unmasked_ce = float("nan") if unmasked_count.sum() <= 0 else float(unmasked_ce_tensor.mean().item())
        exit_gap = (0.0 if masked_count.sum() <= 0 else masked_ce) - (0.0 if unmasked_count.sum() <= 0 else unmasked_ce)

        row = {
            "step": int(step),
            "train_loss_nats": float(loss.item()),
            "masked_ce_nats": masked_ce,
            "unmasked_ce_nats": unmasked_ce,
            "exit_gap_nats": exit_gap,
            "mean_s": float(s_per_sequence.mean().item()),
            "empirical_mask_rate": float(mask_f.mean().item()),
            "forced_mask_sequences": int(forced_count),
            "masked_tokens": int(masked_count.sum().item()),
            "unmasked_tokens": int(unmasked_count.sum().item()),
            "grad_norm": grad_norm,
            "embed_norm": _embedding_norm(model),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history_rows.append(row)

        if step == 1 or step % int(cfg.logging.log_every) == 0 or step == int(cfg.train.n_steps):
            logger.log(
                {
                    "train/loss": row["train_loss_nats"],
                    "train/masked_ce": row["masked_ce_nats"],
                    "train/unmasked_ce": row["unmasked_ce_nats"],
                    "train/exit_gap": row["exit_gap_nats"],
                    "train/mean_s": row["mean_s"],
                    "train/mask_rate": row["empirical_mask_rate"],
                    "train/grad_norm": row["grad_norm"],
                    "train/embed_norm": row["embed_norm"],
                    "train/lr": row["lr"],
                },
                step=step,
            )

        _maybe_log_min_embed_distance(cfg, step=step, model=model, logger=logger)
        _maybe_plot_exit_curve(cfg, step=step, model=model, sampler=sampler, device=device, output_dir=output_dir, logger=logger)
        if _maybe_early_stop(cfg, step=step, model=model, sampler=sampler, device=device, logger=logger):
            break

    eval_metrics, s_grid, model_curve, reference_curve, curve_metrics, curve_rows = _run_exit_evaluation(
        cfg,
        model=model,
        sampler=sampler,
        device=device,
    )

    metrics = {
        "run_id": run_id,
        "family": sampler.family,
        "device": str(device),
        "seed": int(cfg.seed),
        "alphabet_size": int(cfg.data.alphabet_size),
        "seq_len": int(cfg.train.seq_len),
        "n_steps": int(cfg.train.n_steps),
        "batch_size": int(cfg.train.batch_size),
        "eval_batch_size": int(cfg.eval.eval_batch_size),
        "eval_batches": int(cfg.eval.eval_batches),
        "n_s_points": int(cfg.eval.n_s_points),
        "lr": float(cfg.train.lr),
        "grad_clip": grad_clip,
        "model_params": int(model_params),
        "eval_loss_nats_per_token": float(eval_metrics["loss_nats"]),
        "eval_masked_ce_nats_per_masked_token": float(eval_metrics["masked_ce_nats"]),
        "eval_unmasked_ce_nats_per_unmasked_token": float(eval_metrics["unmasked_ce_nats"]),
        "eval_exit_gap_nats": float(eval_metrics["exit_gap_nats"]),
        "eval_mean_s": float(eval_metrics["mean_s"]),
        "eval_empirical_mask_rate": float(eval_metrics["empirical_mask_rate"]),
        "eval_forced_mask_fraction": float(eval_metrics["forced_mask_fraction"]),
        "eval_masked_tokens": int(eval_metrics["masked_tokens"]),
        "eval_unmasked_tokens": int(eval_metrics["unmasked_tokens"]),
        **curve_metrics,
        **sampler.metadata,
    }

    save_csv(output_dir / cfg.paths.history_filename, history_rows)
    save_csv(output_dir / cfg.paths.curve_filename, curve_rows)
    save_json(output_dir / cfg.paths.metrics_filename, metrics)
    export_curve_table(
        output_dir / cfg.paths.tikz_curve_filename,
        [(row["s"], row["model_exit"]) for row in curve_rows],
        x_name="s",
        y_name="exit_gap",
    )
    save_exit_curve_plot(
        output_dir / cfg.paths.curve_plot_filename,
        s_grid=s_grid,
        true_exit=reference_curve["exit_gap"],
        model_exit=model_curve["exit_gap"],
        true_area=curve_metrics["reference_exit_area_nats"],
        model_area=curve_metrics["eval_exit_area_nats"],
        entropy=float(sampler.reference_entropy_nats),
        title=f"EXIT Curves ({sampler.family})",
    )
    torch.save(model.state_dict(), output_dir / cfg.paths.model_filename)

    logger.log({"eval": metrics})
    logger.log_image("eval/exit_curve", output_dir / cfg.paths.curve_plot_filename, caption="final")
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
        "metrics": metrics,
        "family": sampler.family,
    }


def main() -> None:
    config_name, overrides = resolve_config_name_and_overrides(sys.argv[1:])
    cfg = compose_config(PROJECT_ROOT, sys.argv[1:])
    result = run_experiment(cfg, config_name=config_name, overrides=overrides, project_root=PROJECT_ROOT)
    metrics = result["metrics"]
    output_dir = result["output_dir"]
    family = result["family"]
    print(f"[{family}] run_id={result['run_id']}")
    print(
        f"[{family}] done eval_loss={metrics['eval_loss_nats_per_token']:.6f} "
        f"eval_exit_gap={metrics['eval_exit_gap_nats']:.6f} "
        f"exit_area={metrics['eval_exit_area_nats']:.6f} "
        f"reference_entropy={metrics['reference_entropy_nats']:.6f}"
    )
    print(f"Saved outputs under: {output_dir}")


if __name__ == "__main__":
    main()
