"""Generic training loop plus data-aware training dispatch."""

from __future__ import annotations

from collections import deque
import math
from dataclasses import dataclass
from typing import Any, Callable, Literal
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .callbacks import MetricHistory
from .channels import awgn
from .data_common import TrainingBatch
from .models import posterior_mean_from_logits


TrainStopCallback = Callable[[int, torch.nn.Module, MetricHistory, torch.optim.Optimizer], bool]
TrainStepFn = Callable[[torch.nn.Module], dict[str, torch.Tensor]]
ExpectationMethod = Literal["regression", "probs"]


@dataclass(slots=True)
class TrainingRequest:
    model: torch.nn.Module
    data_spec: Any
    device: torch.device | str
    n_steps: int
    lr: float
    grad_accum_steps: int
    grad_clip_embedding: float
    grad_clip_attention: float
    grad_clip_alphabet_proj: float
    grad_clip_other: float
    batch_size: int
    snr_min: float
    snr_max: float
    loss_name: str = "nll"
    label_smoothing: float = 0.0
    estimator: str = "regression"
    topk: int | None = None
    weight_loss_by_snr: bool = True
    logger: object | None = None
    verbose: bool = False
    log_every: int | None = None
    stop_callback: TrainStopCallback | None = None
    seed: int = 0


def _make_progress_bar(total: int, verbose: bool, desc: str):
    if not verbose:
        return None
    return tqdm(total=total, desc=desc, unit="step", leave=True)


class _MovingAverage:
    def __init__(self, window: int) -> None:
        self.values: deque[float] = deque(maxlen=max(1, window))
        self.total = 0.0

    def update(self, value: float) -> float:
        if len(self.values) == self.values.maxlen:
            self.total -= self.values[0]
        self.values.append(value)
        self.total += value
        return self.total / len(self.values)


def train_regression(
    model: torch.nn.Module,
    step_fn: TrainStepFn,
    device: torch.device | str,
    n_steps: int,
    lr: float,
    grad_accum_steps: int,
    grad_clip_embedding: float,
    grad_clip_attention: float,
    grad_clip_alphabet_proj: float,
    grad_clip_other: float,
    logger: object | None = None,
    verbose: bool = False,
    log_every: int | None = None,
    stop_callback: TrainStopCallback | None = None,
    desc: str = "train",
) -> MetricHistory:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = MetricHistory()
    model.train()
    grad_accum_steps = max(1, int(grad_accum_steps))
    log_every = max(1, n_steps // 10) if log_every is None else max(1, int(log_every))
    print_every = None if logger is None else getattr(logger, "print_every", None)
    progress = _make_progress_bar(total=n_steps, verbose=verbose, desc=desc)
    avg_window = min(100, max(10, log_every))
    loss_avg = _MovingAverage(avg_window)
    mse_avg = _MovingAverage(avg_window)
    divergence_avg = _MovingAverage(avg_window)
    nll_avg = _MovingAverage(avg_window)
    nll_ext_avg = _MovingAverage(avg_window)

    try:
        for step in range(1, n_steps + 1):
            optimizer.zero_grad()
            loss = 0.0
            mse = 0.0
            divergence = 0.0
            nll = 0.0
            nll_ext = 0.0
            for _ in range(grad_accum_steps):
                stats = step_fn(model)
                (stats["loss"] / float(grad_accum_steps)).backward()
                loss += float(stats["loss"].item())
                mse += float(stats["mse"].item())
                divergence += float(stats["divergence"].item())
                nll += float(stats["nll"].item())
                nll_ext += float(stats["nll_ext"].item())
            grad_norms = _clip_grad_groups(
                model,
                grad_clip_embedding=grad_clip_embedding,
                grad_clip_attention=grad_clip_attention,
                grad_clip_alphabet_proj=grad_clip_alphabet_proj,
                grad_clip_other=grad_clip_other,
            )
            grad_norm = math.sqrt(
                grad_norms["embedding"] ** 2
                + grad_norms["attention"] ** 2
                + grad_norms["alphabet_proj"] ** 2
                + grad_norms["other"] ** 2
            )
            optimizer.step()

            loss /= float(grad_accum_steps)
            mse /= float(grad_accum_steps)
            divergence /= float(grad_accum_steps)
            nll /= float(grad_accum_steps)
            nll_ext /= float(grad_accum_steps)
            history.append(
                step=step,
                conservation_law=mse,
                mse=mse,
                divergence=divergence,
                nll=nll,
                nll_ext=nll_ext,
                loss=loss,
            )
            loss_ma = loss_avg.update(loss)
            mse_ma = mse_avg.update(mse)
            divergence_ma = divergence_avg.update(divergence)
            nll_ma = nll_avg.update(nll)
            nll_ext_ma = nll_ext_avg.update(nll_ext)
            should_log = logger is not None and (step == 1 or step % log_every == 0 or step == n_steps)
            should_compute_embed_norm = progress is not None or should_log
            embed_norm = None
            if should_compute_embed_norm:
                embed_norm = float(torch.linalg.norm(model.x_values.weight.detach(), dim=1).mean().item())
            if progress is not None:
                progress.update(1)
                progress.set_postfix(
                    loss=f"{loss_ma:.4f}",
                    mse=f"{mse_ma:.4f}",
                    div=f"{divergence_ma:.4f}",
                    nll=f"{nll_ma:.4f}",
                    nll_x=f"{nll_ext_ma:.4f}",
                    grad=f"{grad_norm:.4f}",
                    emb=f"{embed_norm:.4f}",
                )
            if should_log:
                current_lr = float(optimizer.param_groups[0]["lr"])
                logger.log(
                    {
                        "train/loss": loss,
                        "train/conservation_law_nats": mse,
                        "train/mse": mse,
                        "train/divergence": divergence,
                        "train/nll": nll,
                        "train/nll_ext": nll_ext,
                        "train/loss_ma": loss_ma,
                        "train/mse_ma": mse_ma,
                        "train/divergence_ma": divergence_ma,
                        "train/nll_ma": nll_ma,
                        "train/nll_ext_ma": nll_ext_ma,
                        "train/grad_norm": grad_norm,
                        "train/grad_norm_embedding": grad_norms["embedding"],
                        "train/grad_norm_attention": grad_norms["attention"],
                        "train/grad_norm_alphabet_proj": grad_norms["alphabet_proj"],
                        "train/grad_norm_other": grad_norms["other"],
                        "train/embed_norm": embed_norm,
                        "train/lr": current_lr,
                    },
                    step=step,
                )
            if stop_callback is not None and stop_callback(step, model, history, optimizer):
                if progress is not None:
                    progress.set_postfix(
                        loss=f"{loss_ma:.4f}",
                        mse=f"{mse_ma:.4f}",
                        div=f"{divergence_ma:.4f}",
                        nll=f"{nll_ma:.4f}",
                        nll_x=f"{nll_ext_ma:.4f}",
                        grad=f"{grad_norm:.4f}",
                        emb=f"{embed_norm:.4f}",
                        status="stop",
                    )
                break
    finally:
        if progress is not None:
            progress.close()

    model.eval()
    return history


def _clip_grad_norm(parameters: list[torch.nn.Parameter], max_norm: float) -> float:
    trainable = [param for param in parameters if param.requires_grad]
    if not trainable:
        return 0.0
    return float(torch.nn.utils.clip_grad_norm_(trainable, max_norm=max_norm))


def _clip_grad_groups(
    model: torch.nn.Module,
    *,
    grad_clip_embedding: float,
    grad_clip_attention: float,
    grad_clip_alphabet_proj: float,
    grad_clip_other: float,
) -> dict[str, float]:
    embedding_params = [model.x_values.weight]
    attention_params: list[torch.nn.Parameter] = []
    seen_attention_ids: set[int] = set()
    for module in model.modules():
        if not isinstance(module, torch.nn.MultiheadAttention):
            continue
        for param in module.parameters():
            if id(param) in seen_attention_ids:
                continue
            seen_attention_ids.add(id(param))
            attention_params.append(param)
    alphabet_proj_params = list(model.prob_proj.parameters())

    excluded_ids = {id(param) for param in embedding_params}
    excluded_ids.update(id(param) for param in attention_params)
    excluded_ids.update(id(param) for param in alphabet_proj_params)
    other_params = [param for param in model.parameters() if id(param) not in excluded_ids]

    return {
        "embedding": _clip_grad_norm(embedding_params, max_norm=grad_clip_embedding),
        "attention": _clip_grad_norm(attention_params, max_norm=grad_clip_attention),
        "alphabet_proj": _clip_grad_norm(alphabet_proj_params, max_norm=grad_clip_alphabet_proj),
        "other": _clip_grad_norm(other_params, max_norm=grad_clip_other),
    }


def _expectation_target(
    emb_hat: torch.Tensor,
    logits: torch.Tensor,
    x_values: torch.Tensor,
    method: ExpectationMethod,
    topk: int | None = None,
) -> torch.Tensor:
    if method == "regression":
        return emb_hat
    if method == "probs":
        return posterior_mean_from_logits(logits, x_values, topk=topk)
    raise ValueError(f"Unsupported expectation method {method!r}")


def _build_training_step(
    data_spec: Any,
    device: torch.device | str,
    snr_min: float,
    snr_max: float,
    batch_size: int,
    seed: int,
    loss_name: str = "nll",
    label_smoothing: float = 0.0,
    estimator: str = "regression",
    topk: int | None = None,
    weight_loss_by_snr: bool = True,
) -> TrainStepFn:
    log_snr_width = math.log(float(snr_max)) - math.log(float(snr_min))
    sample_batch = data_spec.make_training_batch_sampler(
        device=device,
        batch_size=batch_size,
        snr_min=snr_min,
        snr_max=snr_max,
        seed=seed,
    )
    if loss_name not in {"mse_nll", "mse_nll_ext", "nll_ext", "nll"}:
        raise ValueError(f"Unsupported loss_name={loss_name!r}.")
    include_nll = loss_name == "mse_nll"
    include_mse = loss_name in {"mse_nll", "mse_nll_ext"}
    nll_only = loss_name == "nll"

    def _logits_divergence(logits: torch.Tensor, logits_ext: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=-1)
        log_probs_ext = torch.log_softmax(logits_ext, dim=-1)
        target_idx = states.unsqueeze(-1)
        target_log_probs = torch.gather(log_probs, dim=-1, index=target_idx)
        target_log_probs_ext = torch.gather(log_probs_ext, dim=-1, index=target_idx)
        return target_log_probs - target_log_probs_ext

    def _step(model: torch.nn.Module) -> dict[str, torch.Tensor]:
        batch: TrainingBatch = sample_batch()
        # states: [B] for IID or [B, S] for Markov
        states = batch.states.long()
        # snr: [B, 1] for IID or [B, S, 1] for Markov
        snr = batch.snr
        # clean: [B, D] for IID or [B, S, D] for Markov
        clean = model.x_values(states.long())
        # noisy: same shape as clean
        noisy = awgn(clean, snr)
        # emb_hat: same shape as clean, logits_ext/logits: [B, K] for IID or [B, S, K] for Markov
        emb_hat, logits_ext, logits = model(noisy, snr)

        # weight: [B, 1] for IID or [B, S, 1] for Markov
        weight = snr * log_snr_width
        loss_weight = weight if weight_loss_by_snr else torch.ones_like(weight)
        mse_target = _expectation_target(
            emb_hat=emb_hat,
            logits=logits,
            x_values=model.x_values.weight,
            method="probs" if estimator == "topk" else estimator,
            topk=topk,
        )
        mse_per_token = ((clean.detach() - mse_target) ** 2).sum(dim=-1, keepdim=True) / 2.0
        mse = torch.mean(mse_per_token * weight)
        mse_loss = torch.mean(mse_per_token * loss_weight)
        divergence_per_token = _logits_divergence(logits=logits, logits_ext=logits_ext, states=states)
        divergence = torch.mean(divergence_per_token * weight)
        nll_per_token = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            states.reshape(-1),
            reduction="none",
            label_smoothing=label_smoothing,
        ).reshape(*states.shape, 1)
        nll = torch.mean(nll_per_token * weight)
        nll_loss = torch.mean(nll_per_token * loss_weight)
        nll_ext_per_token = F.cross_entropy(
            logits_ext.reshape(-1, logits_ext.shape[-1]),
            states.reshape(-1),
            reduction="none",
            label_smoothing=label_smoothing,
        ).reshape(*states.shape, 1)
        nll_ext = torch.mean(nll_ext_per_token * weight)
        nll_ext_loss = torch.mean(nll_ext_per_token * loss_weight)
        if include_nll:
            loss = mse_loss + nll_loss + nll_ext_loss
        elif include_mse:
            loss = mse_loss + nll_ext_loss
        elif nll_only:
            loss = nll_loss
        else:
            loss = nll_ext_loss
        return {
            "loss": loss,
            "mse": mse,
            "divergence": divergence,
            "nll": nll,
            "nll_ext": nll_ext,
        }

    return _step


def train_request(request: TrainingRequest) -> MetricHistory:
    if request.loss_name not in {"mse_nll", "mse_nll_ext", "nll_ext", "nll"}:
        raise ValueError(f"Unsupported TrainingRequest.loss_name={request.loss_name!r}.")
    step_fn = _build_training_step(
        data_spec=request.data_spec,
        device=request.device,
        snr_min=request.snr_min,
        snr_max=request.snr_max,
        batch_size=request.batch_size,
        seed=request.seed,
        loss_name=request.loss_name,
        label_smoothing=request.label_smoothing,
        estimator=request.estimator,
        topk=request.topk,
        weight_loss_by_snr=request.weight_loss_by_snr,
    )
    return train_regression(
        model=request.model,
        step_fn=step_fn,
        device=request.device,
        n_steps=request.n_steps,
        lr=request.lr,
        grad_accum_steps=request.grad_accum_steps,
        grad_clip_embedding=request.grad_clip_embedding,
        grad_clip_attention=request.grad_clip_attention,
        grad_clip_alphabet_proj=request.grad_clip_alphabet_proj,
        grad_clip_other=request.grad_clip_other,
        logger=request.logger,
        verbose=request.verbose,
        log_every=request.log_every,
        stop_callback=request.stop_callback,
        desc=f"train:{request.data_spec.run_label}",
    )
