"""Training and evaluation helpers for LM objectives."""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from .bsc import (
    finalize_bsc_metric_accumulator,
    init_bsc_metric_accumulator,
    integrate_bsc_gexit_curve,
    update_bsc_metric_accumulator,
)
from .channels import awgn, apply_ksc_noise, log_ksc_kernel, sample_log_uniform_snr, sample_uniform_ksc_rates
from .lm_distributed import DistributedContext, all_reduce_sum_scalars, unwrap_model
from .exit import (
    apply_sequence_masking,
    finalize_exit_metric_accumulator,
    init_exit_metric_accumulator,
    integrate_exit_curve,
    per_token_nll,
    update_exit_metric_accumulator,
)
from .immse import integrate_half_mmse_curve
from .models import posterior_mean_from_logits


def _log_curve_progress(progress_label: str | None, message: str) -> None:
    if progress_label is not None:
        print(f"[eval-progress] {progress_label}: {message}", flush=True)


def _model_outputs_with_optional_tilt(
    model,
    observed_tokens: torch.Tensor,
    *,
    timesteps: torch.Tensor | None = None,
    channel_tilt: torch.Tensor | None = None,
):
    if hasattr(model, "forward_outputs"):
        try:
            return model.forward_outputs(observed_tokens, timesteps=timesteps, channel_tilt=channel_tilt)
        except ValueError as exc:
            message = str(exc)
            if "without timestep conditioning" not in message:
                raise
            return model.forward_outputs(observed_tokens, timesteps=None, channel_tilt=channel_tilt)
    if timesteps is None:
        logits_ext = model(observed_tokens)
    else:
        try:
            logits_ext = model(observed_tokens, timesteps=timesteps)
        except ValueError as exc:
            message = str(exc)
            if "without timestep conditioning" not in message:
                raise
            logits_ext = model(observed_tokens)
    logits = logits_ext if channel_tilt is None else logits_ext + channel_tilt
    return {"logits_ext": logits_ext, "logits": logits}


def _gaussian_model_posterior_channel_score(
    *,
    logits: torch.Tensor,
    noisy: torch.Tensor,
    x_values: torch.Tensor,
    snr: torch.Tensor,
) -> torch.Tensor:
    posterior_probs = torch.softmax(logits, dim=-1)
    distances = torch.sum((noisy.unsqueeze(2) - x_values.view(1, 1, x_values.shape[0], x_values.shape[1])) ** 2, dim=-1)
    score_all = 0.5 * float(x_values.shape[1]) / snr.squeeze(-1).unsqueeze(-1) - 0.5 * distances
    return torch.sum(posterior_probs * score_all, dim=-1, keepdim=True)


def sample_mask_rates(
    batch_size: int,
    *,
    min_rate: float,
    max_rate: float,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if not 0.0 <= float(min_rate) <= float(max_rate) <= 1.0:
        raise ValueError(f"Mask-rate range must lie in [0, 1], got min={min_rate} max={max_rate}.")
    if generator is None:
        samples = torch.rand((batch_size,), dtype=torch.float32, device=device)
    else:
        samples = torch.rand((batch_size,), generator=generator, dtype=torch.float32).to(device=device)
    return float(min_rate) + (float(max_rate) - float(min_rate)) * samples


def _qsc_variable_t_metrics(
    *,
    losses: torch.Tensor,
    correct_mask: torch.Tensor,
    logits_reweighted: torch.Tensor,
    observed: torch.Tensor,
    t_per_sequence: torch.Tensor,
) -> dict[str, torch.Tensor]:
    t = torch.as_tensor(t_per_sequence, dtype=losses.dtype, device=losses.device).view(-1, 1)
    if t.shape != (losses.shape[0], 1):
        raise ValueError(f"Expected t_per_sequence shape {(losses.shape[0],)}, got {tuple(t_per_sequence.shape)}")
    t = torch.clamp(t, min=torch.finfo(losses.dtype).eps, max=1.0 - torch.finfo(losses.dtype).eps)
    correct_f = correct_mask.to(dtype=losses.dtype)
    incorrect_f = (~correct_mask).to(dtype=losses.dtype)
    log_probs = F.log_softmax(logits_reweighted, dim=-1)
    prob_obs = torch.exp(torch.gather(log_probs, dim=-1, index=observed.unsqueeze(-1)).squeeze(-1))
    score_nll = losses * (correct_f / t - incorrect_f / (1.0 - t))
    mismatch = prob_obs / (t * (1.0 - t)) - 1.0 / (1.0 - t)
    ce_derivative = score_nll + mismatch
    qsc_gexit = -ce_derivative
    token_count = torch.tensor(float(losses.numel()), dtype=losses.dtype, device=losses.device)
    return {
        "score_nll_term_sum": score_nll.sum(),
        "mismatch_correction_sum": mismatch.sum(),
        "ce_derivative_sum": ce_derivative.sum(),
        "qsc_gexit_sum": qsc_gexit.sum(),
        "mean_prob_obs_sum": prob_obs.sum(),
        "token_count": token_count,
    }


def compute_qsc_loss(
    model: torch.nn.Module,
    *,
    tokens: torch.Tensor,
    t: torch.Tensor,
    generator: torch.Generator | None = None,
    label_smoothing: float = 0.0,
) -> dict[str, torch.Tensor]:
    base_model = unwrap_model(model)
    alphabet_size = int(base_model.vocab_size)
    observed_tokens, correct_mask = apply_ksc_noise(tokens, t, alphabet_size, generator=generator)
    kernel = log_ksc_kernel(observed_tokens, t, alphabet_size)
    outputs = _model_outputs_with_optional_tilt(model, observed_tokens, timesteps=t, channel_tilt=kernel)
    logits_ext = outputs["logits_ext"] if isinstance(outputs, dict) else outputs.logits_ext
    logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        tokens.reshape(-1),
        reduction="none",
        label_smoothing=float(label_smoothing),
    ).reshape_as(tokens)
    correct_f = correct_mask.to(dtype=losses.dtype)
    incorrect_f = (~correct_mask).to(dtype=losses.dtype)
    correct_count = torch.clamp(correct_f.sum(), min=1.0)
    incorrect_count = torch.clamp(incorrect_f.sum(), min=1.0)
    accumulator = init_bsc_metric_accumulator()
    update_bsc_metric_accumulator(
        accumulator,
        per_token_loss=losses.detach(),
        correct_mask=correct_mask,
        logits_reweighted=logits.detach(),
        observed=observed_tokens,
        t_per_sequence=t,
    )
    curve_metrics = finalize_bsc_metric_accumulator(accumulator, alphabet_size=alphabet_size)
    return {
        "loss": losses.mean(),
        "losses": losses,
        "observed_tokens": observed_tokens,
        "correct_mask": correct_mask,
        "logits_reweighted": logits,
        "correct_ce": (losses * correct_f).sum() / correct_count,
        "incorrect_ce": (losses * incorrect_f).sum() / incorrect_count,
        "correct_rate": correct_f.mean(),
        "qsc_gexit": torch.tensor(float(curve_metrics["bsc_gexit"]), dtype=losses.dtype, device=losses.device),
        "ce_derivative": torch.tensor(float(curve_metrics["ce_derivative_nats"]), dtype=losses.dtype, device=losses.device),
        "posterior_mismatched": torch.tensor(
            float(curve_metrics["mismatch_correction_nats"]),
            dtype=losses.dtype,
            device=losses.device,
        ),
        "mean_t": torch.tensor(float(curve_metrics["mean_t"]), dtype=losses.dtype, device=losses.device),
    }


def _reduce_bsc_accumulator(
    accumulator: dict[str, float],
    *,
    total_batches: int,
    dist_ctx: DistributedContext,
    device: torch.device | str,
) -> tuple[dict[str, float], int]:
    reduced = all_reduce_sum_scalars(
        dist_ctx,
        [
            accumulator["total_loss_sum"],
            accumulator["total_token_count"],
            accumulator["correct_loss_sum"],
            accumulator["correct_token_count"],
            accumulator["incorrect_loss_sum"],
            accumulator["incorrect_token_count"],
            accumulator["model_prob_obs_sum"],
            accumulator["sequence_count"],
            accumulator["t_sum"],
            float(total_batches),
        ],
        device=device,
    )
    (
        accumulator["total_loss_sum"],
        accumulator["total_token_count"],
        accumulator["correct_loss_sum"],
        accumulator["correct_token_count"],
        accumulator["incorrect_loss_sum"],
        accumulator["incorrect_token_count"],
        accumulator["model_prob_obs_sum"],
        accumulator["sequence_count"],
        accumulator["t_sum"],
        total_batches_f,
    ) = reduced
    return accumulator, int(total_batches_f)


@torch.no_grad()
def evaluate_qsc_objective_metrics(
    model: torch.nn.Module,
    benchmark,
    *,
    split: str,
    batch_size: int,
    device: torch.device | str,
    t_min: float,
    t_max: float,
    max_batches: int | None = None,
    steps_per_batch: int = 1,
    seed: int | None = None,
    dist_ctx: DistributedContext | None = None,
) -> dict[str, float]:
    if dist_ctx is None:
        dist_ctx = DistributedContext(enabled=False)
    model.eval()
    accumulator = init_bsc_metric_accumulator()
    cpu_generator = None
    if seed is not None:
        cpu_generator = torch.Generator(device="cpu")
        cpu_generator.manual_seed(int(seed))
    steps_per_batch = _validated_steps_per_batch(steps_per_batch)

    total_batches = 0
    mc_qsc_gexit_sum = 0.0
    mc_ce_derivative_sum = 0.0
    mc_score_nll_sum = 0.0
    mc_mismatch_sum = 0.0
    mc_prob_obs_sum = 0.0
    mc_token_count = 0.0
    for batch in benchmark.iter_eval_batches(
        split=split,
        batch_size=batch_size,
        device=device,
        include_next_token=False,
        max_batches=max_batches,
        shard_rank=dist_ctx.rank,
        num_shards=dist_ctx.world_size,
    ):
        for _ in range(steps_per_batch):
            t_per_sequence = sample_uniform_ksc_rates(
                (int(batch.shape[0]),),
                t_min=float(t_min),
                t_max=float(t_max),
                device=device,
                generator=cpu_generator,
            )
            stats = compute_qsc_loss(model, tokens=batch, t=t_per_sequence, generator=cpu_generator)
            update_bsc_metric_accumulator(
                accumulator,
                per_token_loss=stats["losses"],
                correct_mask=stats["correct_mask"],
                logits_reweighted=stats["logits_reweighted"],
                observed=stats["observed_tokens"],
                t_per_sequence=t_per_sequence,
            )
            mc = _qsc_variable_t_metrics(
                losses=stats["losses"],
                correct_mask=stats["correct_mask"],
                logits_reweighted=stats["logits_reweighted"],
                observed=stats["observed_tokens"],
                t_per_sequence=t_per_sequence,
            )
            mc_qsc_gexit_sum += float(mc["qsc_gexit_sum"].item())
            mc_ce_derivative_sum += float(mc["ce_derivative_sum"].item())
            mc_score_nll_sum += float(mc["score_nll_term_sum"].item())
            mc_mismatch_sum += float(mc["mismatch_correction_sum"].item())
            mc_prob_obs_sum += float(mc["mean_prob_obs_sum"].item())
            mc_token_count += float(mc["token_count"].item())
            total_batches += 1

    accumulator, total_batches = _reduce_bsc_accumulator(
        accumulator,
        total_batches=total_batches,
        dist_ctx=dist_ctx,
        device=device,
    )
    (
        mc_qsc_gexit_sum,
        mc_ce_derivative_sum,
        mc_score_nll_sum,
        mc_mismatch_sum,
        mc_prob_obs_sum,
        mc_token_count,
    ) = all_reduce_sum_scalars(
        dist_ctx,
        [
            mc_qsc_gexit_sum,
            mc_ce_derivative_sum,
            mc_score_nll_sum,
            mc_mismatch_sum,
            mc_prob_obs_sum,
            mc_token_count,
        ],
        device=device,
    )
    mc_denom = max(float(mc_token_count), 1.0)
    mc_qsc_gexit = float(mc_qsc_gexit_sum / mc_denom)
    mc_ce_derivative = float(mc_ce_derivative_sum / mc_denom)
    mc_score_nll = float(mc_score_nll_sum / mc_denom)
    mc_mismatch = float(mc_mismatch_sum / mc_denom)
    mc_prob_obs = float(mc_prob_obs_sum / mc_denom)
    metrics = finalize_bsc_metric_accumulator(accumulator, int(unwrap_model(model).vocab_size))
    return {
        "loss_nats": float(metrics["loss_nats"]),
        "objective_loss_nats": float(metrics["loss_nats"]),
        "conservation_law_nats": float(mc_qsc_gexit * (float(t_max) - float(t_min))),
        "clean_ce": float(metrics["ce_correct_nats"]),
        "substitute_ce": float(metrics["ce_incorrect_nats"]),
        "posterior_mismatched": mc_mismatch,
        "ce_correct_nats": float(metrics["ce_correct_nats"]),
        "ce_incorrect_nats": float(metrics["ce_incorrect_nats"]),
        "score_nll_term_nats": mc_score_nll,
        "mismatch_correction_nats": mc_mismatch,
        "ce_derivative_nats": mc_ce_derivative,
        "qsc_gexit_nats": mc_qsc_gexit,
        "mean_prob_obs": mc_prob_obs,
        "empirical_correct_rate": float(metrics["empirical_correct_rate"]),
        "mean_t": float(metrics["mean_t"]),
        "evaluated_tokens": int(metrics["total_tokens"]),
        "evaluated_batches": int(total_batches),
        "evaluated_sequences": int(accumulator["sequence_count"]),
    }


@torch.no_grad()
def evaluate_qsc_curve(
    model: torch.nn.Module,
    benchmark,
    *,
    split: str,
    batch_size: int,
    device: torch.device | str,
    t_min: float,
    t_max: float,
    n_t_points: int,
    max_batches: int | None = None,
    steps_per_batch: int = 1,
    seed: int | None = None,
    dist_ctx: DistributedContext | None = None,
    progress_label: str | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float], list[dict[str, float]]]:
    if dist_ctx is None:
        dist_ctx = DistributedContext(enabled=False)
    model.eval()
    t_grid = torch.linspace(float(t_min), float(t_max), steps=int(n_t_points), dtype=torch.float64)
    base_seed = None if seed is None else int(seed)
    steps_per_batch = _validated_steps_per_batch(steps_per_batch)
    curve_rows: list[dict[str, float]] = []
    curve_fields = {
        "loss_nats": [],
        "ce_correct_nats": [],
        "ce_incorrect_nats": [],
        "score_nll_term_nats": [],
        "mismatch_correction_nats": [],
        "ce_derivative_nats": [],
        "qsc_gexit_nats": [],
        "mean_prob_obs": [],
    }

    grid_values = t_grid.tolist()
    for idx, t_value in enumerate(grid_values):
        point_started_at = time.monotonic()
        _log_curve_progress(progress_label, f"starting point {idx + 1}/{len(grid_values)} t={t_value:.6g}")
        local_generator = None
        if base_seed is not None:
            local_generator = torch.Generator(device="cpu")
            local_generator.manual_seed(base_seed + idx)
        accumulator = init_bsc_metric_accumulator()
        total_batches = 0
        for batch in benchmark.iter_eval_batches(
            split=split,
            batch_size=batch_size,
            device=device,
            include_next_token=False,
            max_batches=max_batches,
            shard_rank=dist_ctx.rank,
            num_shards=dist_ctx.world_size,
        ):
            for _ in range(steps_per_batch):
                t_per_sequence = torch.full((int(batch.shape[0]),), float(t_value), dtype=torch.float32, device=device)
                stats = compute_qsc_loss(model, tokens=batch, t=t_per_sequence, generator=local_generator)
                update_bsc_metric_accumulator(
                    accumulator,
                    per_token_loss=stats["losses"],
                    correct_mask=stats["correct_mask"],
                    logits_reweighted=stats["logits_reweighted"],
                    observed=stats["observed_tokens"],
                    t_per_sequence=t_per_sequence,
                )
                total_batches += 1
        accumulator, total_batches = _reduce_bsc_accumulator(
            accumulator,
            total_batches=total_batches,
            dist_ctx=dist_ctx,
            device=device,
        )
        metrics = finalize_bsc_metric_accumulator(accumulator, int(unwrap_model(model).vocab_size))
        _log_curve_progress(
            progress_label,
            f"finished point {idx + 1}/{len(grid_values)} t={t_value:.6g} "
            f"batches={total_batches} elapsed_sec={time.monotonic() - point_started_at:.1f}",
        )
        curve_fields["loss_nats"].append(float(metrics["loss_nats"]))
        curve_fields["ce_correct_nats"].append(float(metrics["ce_correct_nats"]))
        curve_fields["ce_incorrect_nats"].append(float(metrics["ce_incorrect_nats"]))
        curve_fields["score_nll_term_nats"].append(float(metrics["score_nll_term_nats"]))
        curve_fields["mismatch_correction_nats"].append(float(metrics["mismatch_correction_nats"]))
        curve_fields["ce_derivative_nats"].append(float(metrics["ce_derivative_nats"]))
        curve_fields["qsc_gexit_nats"].append(float(metrics["bsc_gexit"]))
        curve_fields["mean_prob_obs"].append(float(metrics["mean_prob_obs"]))
        curve_rows.append(
            {
                "t": float(t_value),
                "loss_nats": float(metrics["loss_nats"]),
                "ce_correct_nats": float(metrics["ce_correct_nats"]),
                "ce_incorrect_nats": float(metrics["ce_incorrect_nats"]),
                "score_nll_term_nats": float(metrics["score_nll_term_nats"]),
                "mismatch_correction_nats": float(metrics["mismatch_correction_nats"]),
                "ce_derivative_nats": float(metrics["ce_derivative_nats"]),
                "qsc_gexit_nats": float(metrics["bsc_gexit"]),
                "mean_prob_obs": float(metrics["mean_prob_obs"]),
            }
        )

    curve_dict = {
        "loss": torch.tensor(curve_fields["loss_nats"], dtype=torch.float64),
        "ce_correct": torch.tensor(curve_fields["ce_correct_nats"], dtype=torch.float64),
        "ce_incorrect": torch.tensor(curve_fields["ce_incorrect_nats"], dtype=torch.float64),
        "score_nll_term": torch.tensor(curve_fields["score_nll_term_nats"], dtype=torch.float64),
        "mismatch_correction": torch.tensor(curve_fields["mismatch_correction_nats"], dtype=torch.float64),
        "ce_derivative": torch.tensor(curve_fields["ce_derivative_nats"], dtype=torch.float64),
        "qsc_gexit": torch.tensor(curve_fields["qsc_gexit_nats"], dtype=torch.float64),
        "mean_prob_obs": torch.tensor(curve_fields["mean_prob_obs"], dtype=torch.float64),
    }
    area = integrate_bsc_gexit_curve(t_grid, curve_dict["qsc_gexit"])
    metrics = {
        "nats_per_token": float(area),
        "n_t_points": int(n_t_points),
        "t_min": float(t_grid.min().item()),
        "t_max": float(t_grid.max().item()),
        "curve_integral_nats": float(area),
    }
    return t_grid, curve_dict, metrics, curve_rows


@torch.no_grad()
def evaluate_causal_nats(
    model: torch.nn.Module,
    benchmark,
    *,
    split: str,
    batch_size: int,
    device: torch.device | str,
    max_batches: int | None = None,
    dist_ctx: DistributedContext | None = None,
) -> dict[str, float]:
    if dist_ctx is None:
        dist_ctx = DistributedContext(enabled=False)
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_sequences = 0
    for batch in benchmark.iter_eval_batches(
        split=split,
        batch_size=batch_size,
        device=device,
        include_next_token=True,
        max_batches=max_batches,
        shard_rank=dist_ctx.rank,
        num_shards=dist_ctx.world_size,
    ):
        logits = model(batch[:, :-1])
        targets = batch[:, 1:]
        loss_sum = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum")
        total_loss += float(loss_sum.item())
        total_tokens += int(targets.numel())
        total_sequences += int(batch.shape[0])
    total_loss, total_tokens_f, total_sequences_f = all_reduce_sum_scalars(
        dist_ctx,
        [float(total_loss), float(total_tokens), float(total_sequences)],
        device=device,
    )
    total_tokens = int(total_tokens_f)
    total_sequences = int(total_sequences_f)
    return {
        "nats_per_token": total_loss / max(total_tokens, 1),
        "evaluated_tokens": int(total_tokens),
        "evaluated_batches": int((total_sequences + int(batch_size) - 1) // int(batch_size)),
    }


def compute_immse_loss(
    model,
    *,
    tokens: torch.Tensor,
    snr: torch.Tensor,
    log_snr_width: float,
    label_smoothing: float = 0.0,
    loss_name: str = "nll",
    weight_loss_by_snr: bool = True,
    high_snr_threshold: float | None = None,
) -> dict[str, torch.Tensor]:
    loss_name = str(loss_name).lower()
    if loss_name not in {"mse_nll", "mse_nll_ext", "nll_ext", "nll"}:
        raise ValueError(f"Unsupported LM IMMSE loss_name={loss_name!r}.")
    base_model = unwrap_model(model)
    clean = base_model.x_values(tokens.long())
    noisy = awgn(clean, snr)
    emb_hat, logits_ext, logits = model(noisy, snr)
    weight = snr * float(log_snr_width)
    loss_weight = weight if bool(weight_loss_by_snr) else torch.ones_like(weight)
    posterior_mean = posterior_mean_from_logits(logits, base_model.x_values.weight.detach())
    mse_per_token = torch.sum((clean.detach() - posterior_mean) ** 2, dim=-1, keepdim=True) / 2.0
    posterior_mismatched_per_token = _gaussian_model_posterior_channel_score(
        logits=logits,
        noisy=noisy.detach(),
        x_values=base_model.x_values.weight.to(dtype=noisy.dtype, device=noisy.device),
        snr=snr,
    )
    mse = torch.mean(mse_per_token * weight)
    mse_loss = torch.mean(mse_per_token * loss_weight)
    if high_snr_threshold is None:
        high_snr_mse_sum = torch.zeros((), dtype=mse_per_token.dtype, device=mse_per_token.device)
        high_snr_token_count = torch.zeros((), dtype=mse_per_token.dtype, device=mse_per_token.device)
    else:
        high_snr_mask = snr >= float(high_snr_threshold)
        high_snr_mse_sum = torch.sum(mse_per_token * high_snr_mask.to(dtype=mse_per_token.dtype))
        high_snr_token_count = torch.sum(high_snr_mask.to(dtype=mse_per_token.dtype))
    posterior_mismatched = torch.mean(posterior_mismatched_per_token)
    nll = torch.mean(
        F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            tokens.reshape(-1),
            reduction="none",
            label_smoothing=float(label_smoothing),
        ).reshape(*tokens.shape, 1)
        * weight
    )
    nll_loss = torch.mean(
        F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            tokens.reshape(-1),
            reduction="none",
            label_smoothing=float(label_smoothing),
        ).reshape(*tokens.shape, 1)
        * loss_weight
    )
    nll_ext = torch.mean(
        F.cross_entropy(
            logits_ext.reshape(-1, logits_ext.shape[-1]),
            tokens.reshape(-1),
            reduction="none",
            label_smoothing=float(label_smoothing),
        ).reshape(*tokens.shape, 1)
        * weight
    )
    nll_ext_loss = torch.mean(
        F.cross_entropy(
            logits_ext.reshape(-1, logits_ext.shape[-1]),
            tokens.reshape(-1),
            reduction="none",
            label_smoothing=float(label_smoothing),
        ).reshape(*tokens.shape, 1)
        * loss_weight
    )
    if loss_name == "mse_nll":
        loss = mse_loss + nll_loss
    elif loss_name == "mse_nll_ext":
        loss = mse_loss + nll_ext_loss
    elif loss_name == "nll":
        loss = nll_loss
    else:
        loss = nll_ext_loss
    return {
        "loss": loss,
        "mse": mse,
        "nll": nll,
        "nll_ext": nll_ext,
        "conservation_law": mse,
        "posterior_mismatched": posterior_mismatched,
        "high_snr_mse_sum": high_snr_mse_sum,
        "high_snr_token_count": high_snr_token_count,
    }


def _sample_immse_snr(
    model,
    *,
    batch_size: int,
    context_length: int,
    snr_min: float,
    snr_max: float,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    base_model = unwrap_model(model)
    if bool(getattr(base_model, "expects_per_image_snr", False)):
        snr = sample_log_uniform_snr(
            (int(batch_size), 1, 1),
            snr_min=snr_min,
            snr_max=snr_max,
            device=device,
            generator=generator,
        )
        return snr.expand(int(batch_size), int(context_length), 1)
    return sample_log_uniform_snr(
        (int(batch_size), int(context_length), 1),
        snr_min=snr_min,
        snr_max=snr_max,
        device=device,
        generator=generator,
    )


@torch.no_grad()
def evaluate_exit_curve(
    model: torch.nn.Module,
    benchmark,
    *,
    split: str,
    batch_size: int,
    device: torch.device | str,
    n_s_points: int,
    max_batches: int | None = None,
    steps_per_batch: int = 1,
    seed: int | None = None,
    dist_ctx: DistributedContext | None = None,
    progress_label: str | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float], list[dict[str, float]]]:
    if dist_ctx is None:
        dist_ctx = DistributedContext(enabled=False)
    model.eval()
    base_model = unwrap_model(model)
    cpu_generator = None
    if seed is not None:
        cpu_generator = torch.Generator(device="cpu")
        cpu_generator.manual_seed(int(seed))
    steps_per_batch = _validated_steps_per_batch(steps_per_batch)
    s_grid = torch.linspace(0.0, 1.0, steps=int(n_s_points), dtype=torch.float64)
    masked_curve: list[float] = []
    unmasked_curve: list[float] = []
    exit_curve: list[float] = []

    grid_values = s_grid.tolist()
    for idx, s in enumerate(grid_values):
        point_started_at = time.monotonic()
        _log_curve_progress(progress_label, f"starting point {idx + 1}/{len(grid_values)} s={s:.6g}")
        local_generator = None
        if cpu_generator is not None:
            local_generator = torch.Generator(device="cpu")
            local_generator.manual_seed(int(seed) + idx)
        masked_sum = 0.0
        masked_count = 0
        unmasked_sum = 0.0
        unmasked_count = 0
        sequence_count = 0
        forced_count = 0

        for batch in benchmark.iter_eval_batches(
            split=split,
            batch_size=batch_size,
            device=device,
            include_next_token=False,
            max_batches=max_batches,
            shard_rank=dist_ctx.rank,
            num_shards=dist_ctx.world_size,
        ):
            for _ in range(steps_per_batch):
                s_per_sequence = torch.full((batch.shape[0],), float(s), dtype=torch.float32, device=device)
                masked_tokens, mask, forced = apply_sequence_masking(
                    batch,
                    s_per_sequence,
                    mask_token_id=base_model.mask_token_id,
                    generator=local_generator,
                )
                logits = model(masked_tokens, timesteps=s_per_sequence)
                losses = per_token_nll(logits, batch)
                mask_f = mask.to(dtype=losses.dtype)
                unmasked_f = (~mask).to(dtype=losses.dtype)
                masked_sum += float((losses * mask_f).sum().item())
                masked_count += int(mask_f.sum().item())
                unmasked_sum += float((losses * unmasked_f).sum().item())
                unmasked_count += int(unmasked_f.sum().item())
                sequence_count += int(batch.shape[0])
                forced_count += int(forced)
        masked_sum, masked_count_f, unmasked_sum, unmasked_count_f, sequence_count_f, forced_count_f = all_reduce_sum_scalars(
            dist_ctx,
            [
                float(masked_sum),
                float(masked_count),
                float(unmasked_sum),
                float(unmasked_count),
                float(sequence_count),
                float(forced_count),
            ],
            device=device,
        )
        masked_count = int(masked_count_f)
        unmasked_count = int(unmasked_count_f)
        sequence_count = int(sequence_count_f)
        forced_count = int(forced_count_f)

        masked_ce = float("nan") if masked_count <= 0 else masked_sum / masked_count
        unmasked_ce = float("nan") if unmasked_count <= 0 else unmasked_sum / unmasked_count
        exit_gap = (0.0 if masked_count <= 0 else masked_ce) - (0.0 if unmasked_count <= 0 else unmasked_ce)
        _log_curve_progress(
            progress_label,
            f"finished point {idx + 1}/{len(grid_values)} s={s:.6g} "
            f"sequences={sequence_count} elapsed_sec={time.monotonic() - point_started_at:.1f}",
        )
        masked_curve.append(masked_ce)
        unmasked_curve.append(unmasked_ce)
        exit_curve.append(exit_gap)

    curve_dict = {
        "masked_ce": torch.tensor(masked_curve, dtype=torch.float64),
        "unmasked_ce": torch.tensor(unmasked_curve, dtype=torch.float64),
        "exit_gap": torch.tensor(exit_curve, dtype=torch.float64),
    }
    area = integrate_exit_curve(s_grid, curve_dict["exit_gap"])
    rows = [
        {
            "s": float(s),
            "masked_ce_nats": float(masked_ce),
            "unmasked_ce_nats": float(unmasked_ce),
            "exit_gap_nats": float(exit_gap),
        }
        for s, masked_ce, unmasked_ce, exit_gap in zip(
            s_grid.tolist(),
            curve_dict["masked_ce"].tolist(),
            curve_dict["unmasked_ce"].tolist(),
            curve_dict["exit_gap"].tolist(),
        )
    ]
    metrics = {
        "nats_per_token": float(area),
        "n_s_points": int(n_s_points),
        "s_min": float(s_grid.min().item()),
        "s_max": float(s_grid.max().item()),
        "curve_integral_nats": float(area),
    }
    return s_grid, curve_dict, metrics, rows


@torch.no_grad()
def evaluate_exit_objective_metrics(
    model: torch.nn.Module,
    benchmark,
    *,
    split: str,
    batch_size: int,
    device: torch.device | str,
    min_rate: float,
    max_rate: float,
    max_batches: int | None = None,
    steps_per_batch: int = 1,
    seed: int | None = None,
    dist_ctx: DistributedContext | None = None,
) -> dict[str, float]:
    if dist_ctx is None:
        dist_ctx = DistributedContext(enabled=False)
    model.eval()
    base_model = unwrap_model(model)
    accumulator = init_exit_metric_accumulator()
    cpu_generator = None
    if seed is not None:
        cpu_generator = torch.Generator(device="cpu")
        cpu_generator.manual_seed(int(seed))
    steps_per_batch = _validated_steps_per_batch(steps_per_batch)

    total_batches = 0
    mc_exit_gap_sum = 0.0
    mc_exit_gap_count = 0.0
    for batch in benchmark.iter_eval_batches(
        split=split,
        batch_size=batch_size,
        device=device,
        include_next_token=False,
        max_batches=max_batches,
        shard_rank=dist_ctx.rank,
        num_shards=dist_ctx.world_size,
    ):
        for _ in range(steps_per_batch):
            s_per_sequence = sample_mask_rates(
                int(batch.shape[0]),
                min_rate=float(min_rate),
                max_rate=float(max_rate),
                device=device,
                generator=cpu_generator,
            )
            masked_tokens, mask, forced_count = apply_sequence_masking(
                batch,
                s_per_sequence,
                mask_token_id=base_model.mask_token_id,
                generator=cpu_generator,
            )
            logits = model(masked_tokens, timesteps=s_per_sequence)
            losses = per_token_nll(logits, batch)
            mask_f = mask.to(dtype=losses.dtype)
            unmasked_f = (~mask).to(dtype=losses.dtype)
            masked_count = torch.clamp(mask_f.sum(dim=1), min=1.0)
            unmasked_count = torch.clamp(unmasked_f.sum(dim=1), min=1.0)
            masked_ce = torch.sum(losses * mask_f, dim=1) / masked_count
            unmasked_ce = torch.sum(losses * unmasked_f, dim=1) / unmasked_count
            mc_exit_gap_sum += float(torch.sum(masked_ce - unmasked_ce).item())
            mc_exit_gap_count += float(batch.shape[0])
            update_exit_metric_accumulator(
                accumulator,
                per_token_loss=losses,
                mask=mask,
                s_per_sequence=s_per_sequence,
                forced_mask_sequences=forced_count,
            )
            total_batches += 1

    reduced = all_reduce_sum_scalars(
        dist_ctx,
        [
            accumulator["total_loss_sum"],
            accumulator["total_token_count"],
            accumulator["masked_loss_sum"],
            accumulator["masked_token_count"],
            accumulator["unmasked_loss_sum"],
            accumulator["unmasked_token_count"],
            accumulator["sequence_count"],
            accumulator["s_sum"],
            accumulator["forced_mask_sequences"],
            float(total_batches),
            mc_exit_gap_sum,
            mc_exit_gap_count,
        ],
        device=device,
    )
    (
        accumulator["total_loss_sum"],
        accumulator["total_token_count"],
        accumulator["masked_loss_sum"],
        accumulator["masked_token_count"],
        accumulator["unmasked_loss_sum"],
        accumulator["unmasked_token_count"],
        accumulator["sequence_count"],
        accumulator["s_sum"],
        accumulator["forced_mask_sequences"],
        total_batches_f,
        mc_exit_gap_sum,
        mc_exit_gap_count,
    ) = reduced
    total_batches = int(total_batches_f)
    metrics = finalize_exit_metric_accumulator(accumulator)
    mc_exit_gap = float(mc_exit_gap_sum / max(float(mc_exit_gap_count), 1.0))
    mc_conservation_law = float((float(max_rate) - float(min_rate)) * mc_exit_gap)
    return {
        "loss_nats": float(metrics["loss_nats"]),
        "objective_loss_nats": float(metrics["masked_ce_nats"] + metrics["unmasked_ce_nats"]),
        "conservation_law_nats": mc_conservation_law,
        "mc_conservation_law_nats": mc_conservation_law,
        "masked_ce_nats": float(metrics["masked_ce_nats"]),
        "unmasked_ce_nats": float(metrics["unmasked_ce_nats"]),
        "exit_gap_nats": mc_exit_gap,
        "aggregate_exit_gap_nats": float(metrics["exit_gap_nats"]),
        "empirical_mask_rate": float(metrics["empirical_mask_rate"]),
        "mean_s": float(metrics["mean_s"]),
        "forced_mask_sequences": int(metrics["forced_mask_sequences"]),
        "forced_mask_fraction": float(metrics["forced_mask_fraction"]),
        "evaluated_tokens": int(metrics["total_tokens"]),
        "evaluated_batches": int(total_batches),
        "evaluated_sequences": int(accumulator["sequence_count"]),
    }


@torch.no_grad()
def evaluate_immse_objective_metrics(
    model,
    benchmark,
    *,
    split: str,
    batch_size: int,
    device: torch.device | str,
    snr_min: float,
    snr_max: float,
    loss_name: str = "nll",
    weight_loss_by_snr: bool = True,
    max_batches: int | None = None,
    steps_per_batch: int = 1,
    seed: int | None = None,
    dist_ctx: DistributedContext | None = None,
) -> dict[str, float]:
    if dist_ctx is None:
        dist_ctx = DistributedContext(enabled=False)
    model.eval()
    log_snr_width = torch.log(torch.tensor(float(snr_max), dtype=torch.float64)).item() - torch.log(
        torch.tensor(float(snr_min), dtype=torch.float64)
    ).item()
    cpu_generator = None
    if seed is not None:
        cpu_generator = torch.Generator(device="cpu")
        cpu_generator.manual_seed(int(seed))
    steps_per_batch = _validated_steps_per_batch(steps_per_batch)

    total_loss = 0.0
    total_mse = 0.0
    total_nll = 0.0
    total_nll_ext = 0.0
    total_posterior_mismatched = 0.0
    total_batches = 0
    total_tokens = 0
    total_sequences = 0

    for batch in benchmark.iter_eval_batches(
        split=split,
        batch_size=batch_size,
        device=device,
        include_next_token=False,
        max_batches=max_batches,
        shard_rank=dist_ctx.rank,
        num_shards=dist_ctx.world_size,
    ):
        for _ in range(steps_per_batch):
            snr = _sample_immse_snr(
                model,
                batch_size=int(batch.shape[0]),
                context_length=int(benchmark.context_length),
                snr_min=float(snr_min),
                snr_max=float(snr_max),
                device=device,
                generator=cpu_generator,
            )
            stats = compute_immse_loss(
                model,
                tokens=batch,
                snr=snr,
                log_snr_width=float(log_snr_width),
                loss_name=loss_name,
                weight_loss_by_snr=weight_loss_by_snr,
            )
            total_loss += float(stats["loss"].item())
            total_mse += float(stats["mse"].item())
            total_nll += float(stats["nll"].item())
            total_nll_ext += float(stats["nll_ext"].item())
            total_posterior_mismatched += float(stats["posterior_mismatched"].item())
            total_batches += 1
            total_tokens += int(batch.numel())
            total_sequences += int(batch.shape[0])

    total_loss, total_mse, total_nll, total_nll_ext, total_posterior_mismatched, total_batches_f, total_tokens_f, total_sequences_f = all_reduce_sum_scalars(
        dist_ctx,
        [
            total_loss,
            total_mse,
            total_nll,
            total_nll_ext,
            total_posterior_mismatched,
            float(total_batches),
            float(total_tokens),
            float(total_sequences),
        ],
        device=device,
    )
    total_batches = int(total_batches_f)
    total_tokens = int(total_tokens_f)
    total_sequences = int(total_sequences_f)
    denom = max(total_batches, 1)
    mc_conservation_law = float(total_mse / denom)
    return {
        "objective_loss_nats": float(total_loss / denom),
        "loss_nats": float(total_loss / denom),
        "conservation_law_nats": mc_conservation_law,
        "mc_conservation_law_nats": mc_conservation_law,
        "mse": mc_conservation_law,
        "posterior_mismatched": float(total_posterior_mismatched / denom),
        "nll": float(total_nll / denom),
        "nll_ext": float(total_nll_ext / denom),
        "evaluated_batches": int(total_batches),
        "evaluated_tokens": int(total_tokens),
        "evaluated_sequences": int(total_sequences),
        "snr_min": float(snr_min),
        "snr_max": float(snr_max),
    }


@torch.no_grad()
def evaluate_immse_curve(
    model,
    benchmark,
    *,
    split: str,
    batch_size: int,
    device: torch.device | str,
    snr_min: float,
    snr_max: float,
    n_snr_points: int,
    max_batches: int | None = None,
    steps_per_batch: int = 1,
    seed: int | None = None,
    dist_ctx: DistributedContext | None = None,
    progress_label: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float], list[dict[str, float]]]:
    if dist_ctx is None:
        dist_ctx = DistributedContext(enabled=False)
    model.eval()
    base_model = unwrap_model(model)
    if float(snr_min) <= 0.0 or float(snr_max) <= 0.0:
        raise ValueError("snr_min and snr_max must be strictly positive for I-MMSE evaluation.")
    snr_grid = torch.logspace(
        torch.log10(torch.tensor(float(snr_min), dtype=torch.float64)),
        torch.log10(torch.tensor(float(snr_max), dtype=torch.float64)),
        steps=int(n_snr_points),
        dtype=torch.float64,
    )
    base_seed = None if seed is None else int(seed)
    steps_per_batch = _validated_steps_per_batch(steps_per_batch)
    mmse_values: list[float] = []

    grid_values = snr_grid.tolist()
    for idx, snr_value in enumerate(grid_values):
        point_started_at = time.monotonic()
        _log_curve_progress(progress_label, f"starting point {idx + 1}/{len(grid_values)} snr={snr_value:.6g}")
        if base_seed is not None:
            torch.manual_seed(base_seed + idx)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(base_seed + idx)
        total_sq_error = 0.0
        total_tokens = 0
        for batch in benchmark.iter_eval_batches(
            split=split,
            batch_size=batch_size,
            device=device,
            include_next_token=False,
            max_batches=max_batches,
            shard_rank=dist_ctx.rank,
            num_shards=dist_ctx.world_size,
        ):
            clean = base_model.x_values(batch.long())
            snr = torch.full((*batch.shape, 1), float(snr_value), dtype=clean.dtype, device=device)
            for _ in range(steps_per_batch):
                noisy = awgn(clean, snr)
                _, _, logits = model(noisy, snr)
                posterior_mean = posterior_mean_from_logits(logits, base_model.x_values.weight)
                total_sq_error += float(torch.sum((clean - posterior_mean) ** 2).item())
                total_tokens += int(batch.numel())
        total_sq_error, total_tokens_f = all_reduce_sum_scalars(
            dist_ctx,
            [float(total_sq_error), float(total_tokens)],
            device=device,
        )
        total_tokens = int(total_tokens_f)
        _log_curve_progress(
            progress_label,
            f"finished point {idx + 1}/{len(grid_values)} snr={snr_value:.6g} "
            f"tokens={total_tokens} elapsed_sec={time.monotonic() - point_started_at:.1f}",
        )
        mmse_values.append(total_sq_error / max(total_tokens, 1))

    mmse_curve = torch.tensor(mmse_values, dtype=torch.float64)
    area = integrate_half_mmse_curve(snr_grid, mmse_curve)
    rows = [
        {
            "snr": float(snr),
            "mmse": float(mmse),
            "half_mmse": float(mmse / 2.0),
        }
        for snr, mmse in zip(snr_grid.tolist(), mmse_curve.tolist())
    ]
    metrics = {
        "nats_per_token": float(area),
        "n_snr_points": int(n_snr_points),
        "snr_min": float(snr_grid.min().item()),
        "snr_max": float(snr_grid.max().item()),
        "curve_integral_nats": float(area),
    }
    return snr_grid, mmse_curve, metrics, rows


def _save_single_curve_plot(
    output_path: Path,
    *,
    x_values: torch.Tensor,
    y_values: torch.Tensor,
    title: str,
    area: float,
    xlabel: str,
    ylabel: str,
    xscale: str = "linear",
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x_values_t = torch.as_tensor(x_values, dtype=torch.float64).cpu()
    y_values_t = torch.as_tensor(y_values, dtype=torch.float64).cpu()
    ax.plot(x_values_t, y_values_t, linewidth=2.2, color="tab:orange", label=f"area={area:.4f}")
    ax.fill_between(x_values_t, torch.zeros_like(x_values_t), y_values_t, color="tab:orange", alpha=0.2)
    ax.set_xlim(float(x_values_t.min().item()), float(x_values_t.max().item()))
    ax.set_xscale(xscale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _validated_steps_per_batch(steps_per_batch: int) -> int:
    steps = int(steps_per_batch)
    if steps <= 0:
        raise ValueError(f"steps_per_batch must be positive, got {steps_per_batch}.")
    return steps


def save_exit_curve_plot(output_path: Path, *, s_grid: torch.Tensor, exit_curve: torch.Tensor, title: str, area: float) -> None:
    _save_single_curve_plot(
        output_path,
        x_values=s_grid,
        y_values=exit_curve,
        title=title,
        area=area,
        xlabel="Mask rate s",
        ylabel="masked CE - unmasked CE",
    )


def save_qsc_curve_plot(output_path: Path, *, t_grid: torch.Tensor, qsc_curve: torch.Tensor, title: str, area: float) -> None:
    _save_single_curve_plot(
        output_path,
        x_values=t_grid,
        y_values=qsc_curve,
        title=title,
        area=area,
        xlabel="Pass-through rate t",
        ylabel="QSC GEXIT",
    )


def save_immse_curve_plot(output_path: Path, *, snr_grid: torch.Tensor, mmse_curve: torch.Tensor, title: str, area: float) -> None:
    _save_single_curve_plot(
        output_path,
        x_values=snr_grid,
        y_values=0.5 * torch.as_tensor(mmse_curve, dtype=torch.float64),
        title=title,
        area=area,
        xlabel="SNR gamma",
        ylabel="0.5 * MMSE(gamma)",
        xscale="log",
    )
