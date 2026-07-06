"""Training script for the BSC-variant (K-ary Symmetric Channel) experiment.

Channel model
-------------
  W_t(x | x) = t               ("input is out w.p. t")
  W_t(u | x) = (1-t)/(K-1)    ("w.p. 1-t uniform over the rest")

Training objective
------------------
NLL under the K-SC posterior:

    -log P^θ(X_i | Y)   where  P^θ(X_i | Y) ∝ P^θ(X_i | Y_{-i}) * W_t(Y_i | X_i)

i.e. the model's extrinsic logits are reweighted by log W_t(y_i | x) before
softmax.  The model receives the noisy observed tokens Y as input; no mask
token is needed.

Evaluation (mismatched GEXIT derivative, eq. 5 in the paper)
-------------------------------------------------------------
At the end of training we sweep t over a grid in [1/K, 1) and estimate
the theorem-1 / equation-(5) derivative terms

    d CE / dt = score_nll_term(t) + mismatch_correction(t)

where for the K-SC

    score_nll_term(t) = CE_correct(t) - CE_incorrect(t)
    mismatch_correction(t) = E[P^θ(Y_i|Y)] / (t*(1-t)) - 1/(1-t).

We save the per-t equation-(5) terms and summarize them over the eval grid.

Usage
-----
  python scripts/train_bsc.py --config-name experiments/markov_channel_k128/qsc
  python scripts/train_bsc.py --config-name experiments/markov_channel_k128/qsc data.alphabet_size=64
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "gexit-diffusion-matplotlib"))

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common.config import compose_config, resolve_config_name_and_overrides
from experiments.common.experiment_registry import require_experiment_kind
from experiments.common.io import deterministic_output_dir, save_config, save_csv, save_json
from experiments.common.repro import configure_environment, get_best_device, make_run_id, set_global_seed
from experiments.common.tikz_export import export_curve_table
from experiments.common.wandb_logger import WandbLogger

from src.bsc import (
    evaluate_model_bsc,
    evaluate_model_bsc_curve,
    evaluate_reference_iid_bsc_curve,
    evaluate_reference_markov_bsc_curve,
    integrate_bsc_gexit_curve,
    per_token_nll_bsc,
    save_bsc_curve_plot,
)
from src.channels import apply_ksc_noise, log_ksc_kernel, sample_uniform_ksc_rates
from src.models import BidirectionalMaskedTokenModel
from src.sources_iid import entropy_nats, sample_dirichlet_iid_probs, uniform_iid_probs
from src.sources_markov import build_markov_chain, sample_markov_sequences_torch


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _validate_config(cfg: DictConfig) -> None:
    require_experiment_kind(cfg, "qsc")
    family = str(cfg.family)
    if family not in {"iid", "markov"}:
        raise ValueError(f"bsc experiment only supports family='iid' or 'markov', got {family!r}.")
    if int(cfg.data.alphabet_size) < 2:
        raise ValueError(f"data.alphabet_size must be >= 2, got {cfg.data.alphabet_size}.")
    if int(cfg.train.seq_len) <= 0:
        raise ValueError(f"train.seq_len must be positive, got {cfg.train.seq_len}.")
    if int(cfg.train.n_steps) <= 0:
        raise ValueError(f"train.n_steps must be positive, got {cfg.train.n_steps}.")
    if int(cfg.train.batch_size) <= 0:
        raise ValueError(f"train.batch_size must be positive, got {cfg.train.batch_size}.")
    if float(cfg.train.lr) <= 0.0:
        raise ValueError(f"train.lr must be positive, got {cfg.train.lr}.")
    eval_every = getattr(cfg.eval, "eval_every", None)
    if eval_every is not None and int(eval_every) <= 0:
        raise ValueError(f"eval.eval_every must be positive when set, got {eval_every}.")


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


def _build_model(cfg: DictConfig) -> BidirectionalMaskedTokenModel:
    return BidirectionalMaskedTokenModel(**_model_kwargs_from_cfg(cfg))


# ---------------------------------------------------------------------------
# Data sampler helpers
# ---------------------------------------------------------------------------


def _make_iid_samplers(cfg: DictConfig):
    alphabet_size = int(cfg.data.alphabet_size)
    source_distribution = str(cfg.data.source_distribution)
    seed = int(cfg.seed)
    dirichlet_alpha = cfg.data.get("dirichlet_alpha")

    if source_distribution == "uniform":
        probs = uniform_iid_probs(alphabet_size)
    elif source_distribution == "dirichlet":
        if dirichlet_alpha is None:
            raise ValueError("data.dirichlet_alpha must be set for dirichlet source.")
        probs = sample_dirichlet_iid_probs(alphabet_size, float(dirichlet_alpha), seed)
    else:
        raise ValueError(f"Unsupported source_distribution={source_distribution!r}.")

    probs_cpu = torch.as_tensor(probs, dtype=torch.float32, device="cpu")
    train_gen = torch.Generator(device="cpu")
    train_gen.manual_seed(seed + 101)
    eval_gen = torch.Generator(device="cpu")
    eval_gen.manual_seed(seed + 102)

    def _sample(gen, batch_size, seq_len, device):
        return torch.multinomial(
            probs_cpu,
            num_samples=batch_size * seq_len,
            replacement=True,
            generator=gen,
        ).view(batch_size, seq_len).to(device=device, dtype=torch.long)

    return (
        lambda bs, sl, dev: _sample(train_gen, bs, sl, dev),
        lambda bs, sl, dev: _sample(eval_gen, bs, sl, dev),
        torch.as_tensor(probs, dtype=torch.float64),
        float(entropy_nats(probs)),
    )


def _make_markov_samplers(cfg: DictConfig):
    from src.sources_markov import MarkovChainSpec

    alphabet_size = int(cfg.data.alphabet_size)
    seed = int(cfg.seed)
    dirichlet_alpha = float(cfg.data.dirichlet_alpha)
    init_distribution = str(getattr(cfg.data, "init_distribution", "stationary"))

    chain = build_markov_chain(
        alphabet_size=alphabet_size,
        dirichlet_alpha=dirichlet_alpha,
        seed=seed,
        init_distribution=init_distribution,
    )
    init_probs = torch.as_tensor(chain.init_probs, dtype=torch.float32)
    transition = torch.as_tensor(chain.transition_matrix, dtype=torch.float32)

    train_gen = torch.Generator(device="cpu")
    train_gen.manual_seed(seed + 201)
    eval_gen = torch.Generator(device="cpu")
    eval_gen.manual_seed(seed + 202)

    def _sample(gen, batch_size, seq_len, device):
        uniforms = torch.rand(batch_size, seq_len, generator=gen, dtype=torch.float32)
        states = sample_markov_sequences_torch(
            num_sequences=batch_size,
            seq_len=seq_len,
            init_probs=init_probs,
            transition_matrix=transition,
            uniforms=uniforms,
        )
        return states.to(device=device, dtype=torch.long)

    return (
        lambda bs, sl, dev: _sample(train_gen, bs, sl, dev),
        lambda bs, sl, dev: _sample(eval_gen, bs, sl, dev),
        None,
        float(chain.entropy_rate_nats),
        chain,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _count_parameters(model: torch.nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters())


def run_experiment(cfg: DictConfig, *, project_root: Path = PROJECT_ROOT) -> dict[str, object]:
    require_experiment_kind(cfg, "qsc")
    configure_environment()
    cfg.paths.results_root = str((Path(project_root) / str(cfg.paths.results_root)).resolve())
    _validate_config(cfg)
    set_global_seed(cfg.seed, deterministic=cfg.deterministic)
    device = get_best_device(cfg.device)

    family = str(cfg.family)
    K = int(cfg.data.alphabet_size)
    seq_len = int(cfg.train.seq_len)
    batch_size = int(cfg.train.batch_size)
    n_steps = int(cfg.train.n_steps)
    lr = float(cfg.train.lr)
    grad_clip = float(cfg.train.grad_clip.get("other", 1.0))
    eval_every = getattr(cfg.eval, "eval_every", None)
    eval_every = 0 if eval_every is None else int(eval_every)

    # t range for training: informative regime [1/K, 1]
    t_min_train = 1.0 / K
    t_max_train = 1.0 - 1e-4  # avoid t=1 (perfect channel, trivial gradient)

    # Build samplers
    source_probs = None
    chain_spec = None
    if family == "iid":
        sample_train, sample_eval, source_probs, reference_entropy = _make_iid_samplers(cfg)
    else:
        result = _make_markov_samplers(cfg)
        sample_train, sample_eval, source_probs, reference_entropy, chain_spec = result

    # Build model, optimizer
    model = _build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model_params = _count_parameters(model)

    # Unique run ID and output directory
    config_name, overrides = resolve_config_name_and_overrides(sys.argv[1:])
    run_id = make_run_id(config_name, overrides)
    output_dir = deterministic_output_dir(project_root, cfg.paths.results_root, cfg.name, run_id)
    save_config(output_dir / cfg.paths.config_snapshot_name, cfg)

    logger = WandbLogger(cfg=cfg, run_id=run_id, output_dir=str(output_dir / cfg.paths.wandb_dir_name))
    logger.log({"model/parameter_count": model_params})

    if cfg.verbose:
        print(f"[setup] BSC experiment  family={family}  K={K}")
        print(f"[setup] run_id={run_id}  device={device}")
        print(f"[setup] t_train=[{t_min_train:.4f}, {t_max_train:.4f}]")
        print(f"[setup] reference_entropy={reference_entropy:.6f} nats")
        print("[setup] resolved config:")
        print(OmegaConf.to_yaml(cfg, resolve=True).strip())

    # RNG for noise sampling (CPU, seeded for reproducibility)
    noise_gen = torch.Generator(device="cpu")
    noise_gen.manual_seed(int(cfg.seed) + 303)

    log_every = max(1, getattr(cfg.logging, "log_every", n_steps // 10))
    history_rows: list[dict] = []

    def _finite_mean(values: torch.Tensor) -> float:
        values_t = torch.as_tensor(values, dtype=torch.float64)
        valid = torch.isfinite(values_t)
        if int(valid.sum().item()) == 0:
            return float("nan")
        return float(values_t[valid].mean().item())

    def _finite_std(values: torch.Tensor) -> float:
        values_t = torch.as_tensor(values, dtype=torch.float64)
        valid = torch.isfinite(values_t)
        if int(valid.sum().item()) == 0:
            return float("nan")
        return float(values_t[valid].std(unbiased=False).item())

    def _pairwise_curve_error(model_values: torch.Tensor, ref_values: torch.Tensor, *, kind: str) -> float:
        model_t = torch.as_tensor(model_values, dtype=torch.float64)
        ref_t = torch.as_tensor(ref_values, dtype=torch.float64)
        valid = torch.isfinite(model_t) & torch.isfinite(ref_t)
        if int(valid.sum().item()) == 0:
            return float("nan")
        diff = model_t[valid] - ref_t[valid]
        if kind == "mae":
            return float(diff.abs().mean().item())
        if kind == "rmse":
            return float(torch.sqrt(torch.mean(diff.square())).item())
        raise ValueError(f"Unsupported error kind={kind!r}.")

    def _run_bsc_eval(*, step: int | None = None) -> tuple[dict[str, object], dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        n_s_points = int(cfg.eval.n_s_points)
        eval_batches = int(cfg.eval.eval_batches)
        eval_batch_size = int(cfg.eval.eval_batch_size)

        total_eval_tokens = eval_batch_size * seq_len * eval_batches
        t_min_eval = 1.0 / K + 1e-3
        t_max_eval = min(1.0 - 20.0 / max(total_eval_tokens, 1), 1.0 - 1e-3)
        t_max_eval = max(t_max_eval, t_min_eval + 0.05)
        t_grid = torch.linspace(t_min_eval, t_max_eval, steps=n_s_points, dtype=torch.float64)

        model_curves = evaluate_model_bsc_curve(
            model,
            sample_eval,
            t_grid=t_grid,
            batch_size=eval_batch_size,
            seq_len=seq_len,
            eval_batches=eval_batches,
            alphabet_size=K,
            device=device,
            seed=int(cfg.seed) + 10_000 + (0 if step is None else int(step) * 100),
            steps_per_batch=int(getattr(cfg.eval, "steps_per_batch", 1)),
        )

        if family == "iid" and source_probs is not None:
            reference_curves = evaluate_reference_iid_bsc_curve(
                source_probs,
                t_grid=t_grid,
                alphabet_size=K,
                batch_size=eval_batch_size,
                seq_len=seq_len,
                eval_batches=eval_batches,
                seed=int(cfg.seed) + 20_000 + (0 if step is None else int(step) * 100),
            )
        elif family == "markov" and chain_spec is not None:
            reference_curves = evaluate_reference_markov_bsc_curve(
                sample_eval,
                init_probs=chain_spec.init_probs,
                transition_matrix=chain_spec.transition_matrix,
                t_grid=t_grid,
                alphabet_size=K,
                batch_size=eval_batch_size,
                seq_len=seq_len,
                eval_batches=eval_batches,
                seed=int(cfg.seed) + 20_000 + (0 if step is None else int(step) * 100),
            )
        else:
            nan_curve = torch.full_like(t_grid, float("nan"))
            reference_curves = {
                "ce_correct": nan_curve,
                "ce_incorrect": nan_curve,
                "score_nll_term": nan_curve,
                "mismatch_correction": nan_curve,
                "ce_derivative": nan_curve,
                "bsc_gexit": nan_curve,
            }

        model_area = integrate_bsc_gexit_curve(t_grid, model_curves["bsc_gexit"])
        reference_area = integrate_bsc_gexit_curve(t_grid, reference_curves["bsc_gexit"])
        model_area_abs_error = abs(model_area - reference_entropy)
        model_area_rel_error = model_area_abs_error / max(abs(reference_entropy), 1e-12)
        reference_area_abs_error = abs(reference_area - reference_entropy)
        reference_area_rel_error = reference_area_abs_error / max(abs(reference_entropy), 1e-12)

        metrics = {
            "reference_entropy_nats": float(reference_entropy),
            "eval_t_min": float(t_grid.min().item()),
            "eval_t_max": float(t_grid.max().item()),
            "eval_n_t_points": int(t_grid.numel()),
            "model_eq5_ce_derivative_mean_nats": _finite_mean(model_curves["ce_derivative"]),
            "model_eq5_ce_derivative_std_nats": _finite_std(model_curves["ce_derivative"]),
            "model_eq5_score_nll_term_mean_nats": _finite_mean(model_curves["score_nll_term"]),
            "model_eq5_mismatch_correction_mean_nats": _finite_mean(model_curves["mismatch_correction"]),
            "reference_eq5_ce_derivative_mean_nats": _finite_mean(reference_curves["ce_derivative"]),
            "reference_eq5_score_nll_term_mean_nats": _finite_mean(reference_curves["score_nll_term"]),
            "reference_eq5_mismatch_correction_mean_nats": _finite_mean(reference_curves["mismatch_correction"]),
            "eq5_ce_derivative_mae_nats": _pairwise_curve_error(
                model_curves["ce_derivative"], reference_curves["ce_derivative"], kind="mae"
            ),
            "eq5_ce_derivative_rmse_nats": _pairwise_curve_error(
                model_curves["ce_derivative"], reference_curves["ce_derivative"], kind="rmse"
            ),
            "eq5_score_nll_term_mae_nats": _pairwise_curve_error(
                model_curves["score_nll_term"], reference_curves["score_nll_term"], kind="mae"
            ),
            "eq5_mismatch_correction_mae_nats": _pairwise_curve_error(
                model_curves["mismatch_correction"], reference_curves["mismatch_correction"], kind="mae"
            ),
            "model_ce_area_nats": float(model_area),
            "reference_ce_area_nats": float(reference_area),
            "model_ce_area_abs_error": float(model_area_abs_error),
            "model_ce_area_rel_error": float(model_area_rel_error),
            "reference_ce_area_abs_error": float(reference_area_abs_error),
            "reference_ce_area_rel_error": float(reference_area_rel_error),
        }
        return metrics, model_curves, reference_curves, t_grid

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    from tqdm.auto import tqdm

    pbar = tqdm(total=n_steps, desc="train", unit="step", leave=True) if cfg.verbose else None

    for step in range(1, n_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        tokens = sample_train(batch_size, seq_len, device).long()  # [B, S]

        # Sample per-sequence pass-through rate t ~ Uniform[t_min, t_max]
        t_per_seq = sample_uniform_ksc_rates(
            (batch_size,), t_min_train, t_max_train, device=device, generator=noise_gen
        )  # [B]

        # Apply K-SC noise
        noisy_tokens, correct_mask = apply_ksc_noise(
            tokens, t_per_seq, K, generator=noise_gen
        )  # [B, S], [B, S]

        # Forward pass: model receives noisy tokens and the channel kernel tilt.
        kernel = log_ksc_kernel(noisy_tokens, t_per_seq, K)
        if hasattr(model, "forward_outputs"):
            outputs = model.forward_outputs(noisy_tokens, channel_tilt=kernel)
            logits_ext = outputs.logits_ext
            logits = outputs.logits
        else:
            logits_ext = model(noisy_tokens)  # [B, S, K]
            logits = logits_ext + kernel

        # Per-token NLL = -log P^θ(X_i | Y) with shared logit tilting.
        losses = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            tokens.reshape(-1),
            reduction="none",
        ).reshape_as(tokens)

        loss = losses.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        loss_val = float(loss.item())
        correct_f = correct_mask.float()
        incorrect_f = (~correct_mask).float()
        n_corr = float(correct_f.sum().item())
        n_incorr = float(incorrect_f.sum().item())
        ce_corr = float("nan") if n_corr == 0 else float((losses.detach() * correct_f).sum().item() / n_corr)
        ce_incorr = float("nan") if n_incorr == 0 else float((losses.detach() * incorrect_f).sum().item() / n_incorr)

        row = {
            "step": step,
            "loss": loss_val,
            "ce_correct": ce_corr,
            "ce_incorrect": ce_incorr,
            "mean_t": float(t_per_seq.mean().item()),
        }
        history_rows.append(row)

        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss_val:.4f}", ce_c=f"{ce_corr:.4f}", ce_i=f"{ce_incorr:.4f}")

        should_log = step == 1 or step % log_every == 0 or step == n_steps
        if should_log and logger is not None:
            logger.log({"train/loss": loss_val, "train/ce_correct": ce_corr, "train/ce_incorrect": ce_incorr}, step=step)

        should_eval = eval_every > 0 and step < n_steps and step % eval_every == 0
        if should_eval:
            eval_metrics, _, _, _ = _run_bsc_eval(step=step)
            logger.log({"monitor": eval_metrics}, step=step)

    if pbar is not None:
        pbar.close()

    # -----------------------------------------------------------------------
    # Final evaluation: equation-(5) derivative sweep
    # -----------------------------------------------------------------------
    if cfg.verbose:
        print("[eval] computing equation-(5) curve ...")
    metrics, model_curves, reference_curves, t_grid = _run_bsc_eval(step=n_steps)
    metrics = {
        "run_id": run_id,
        "experiment_kind": str(cfg.kind),
        "objective": str(cfg.objective),
        "family": str(cfg.family),
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
        "model_params": int(model_params),
        **metrics,
    }
    logger.log({"eval": metrics})

    # -----------------------------------------------------------------------
    # Save artefacts
    # -----------------------------------------------------------------------
    curve_rows = [
        {
            "t": float(t),
            "model_ce_correct": float(mc),
            "model_ce_incorrect": float(mi),
            "model_score_nll_term": float(ms),
            "model_mismatch_correction": float(mm),
            "model_ce_derivative": float(md),
            "model_bsc_gexit": float(mg),
            "ref_ce_correct": float(rc),
            "ref_ce_incorrect": float(ri),
            "ref_score_nll_term": float(rs),
            "ref_mismatch_correction": float(rm),
            "ref_ce_derivative": float(rd),
            "ref_bsc_gexit": float(rg),
        }
        for t, mc, mi, ms, mm, md, mg, rc, ri, rs, rm, rd, rg in zip(
            t_grid.tolist(),
            model_curves["ce_correct"].tolist(),
            model_curves["ce_incorrect"].tolist(),
            model_curves["score_nll_term"].tolist(),
            model_curves["mismatch_correction"].tolist(),
            model_curves["ce_derivative"].tolist(),
            model_curves["bsc_gexit"].tolist(),
            reference_curves["ce_correct"].tolist(),
            reference_curves["ce_incorrect"].tolist(),
            reference_curves["score_nll_term"].tolist(),
            reference_curves["mismatch_correction"].tolist(),
            reference_curves["ce_derivative"].tolist(),
            reference_curves["bsc_gexit"].tolist(),
        )
    ]

    save_csv(output_dir / cfg.paths.history_filename, history_rows)
    save_csv(output_dir / cfg.paths.curve_filename, curve_rows)
    save_json(output_dir / cfg.paths.metrics_filename, metrics)
    export_curve_table(
        output_dir / cfg.paths.tikz_curve_filename,
        [(row["t"], row["model_ce_derivative"]) for row in curve_rows],
        x_name="t",
        y_name="ce_derivative",
    )
    save_bsc_curve_plot(
        output_dir / cfg.paths.curve_plot_filename,
        t_grid=t_grid,
        true_bsc_gexit=reference_curves["bsc_gexit"],
        model_bsc_gexit=model_curves["bsc_gexit"],
        true_area=metrics["reference_ce_area_nats"],
        model_area=metrics["model_ce_area_nats"],
        entropy=float(reference_entropy),
        title=f"BSC GEXIT Curves ({family})",
    )
    torch.save(model.state_dict(), output_dir / cfg.paths.model_filename)

    logger.log_image("eval/bsc_gexit_curve", output_dir / cfg.paths.curve_plot_filename, caption="final")
    logger.log_directory_files(
        output_dir,
        artifact_name=f"{run_id}-outputs",
        exclude_dirs=[str(cfg.paths.wandb_dir_name)],
        exclude_files=[str(cfg.paths.curve_plot_filename)],
    )
    logger.finish()

    if cfg.verbose:
        print(f"\nBSC experiment complete  family={family}  K={K}")
        print(f"Reference entropy:       {reference_entropy:.6f} nats")
        print(f"Model CE area (eq. 6):   {metrics['model_ce_area_nats']:.6f}  "
              f"rel={metrics['model_ce_area_rel_error']:.2%}")
        print(f"Model eq. (5) dCE/dt:    mean={metrics['model_eq5_ce_derivative_mean_nats']:.6f}  "
              f"std={metrics['model_eq5_ce_derivative_std_nats']:.6f}")
        print(f"Eq. (5) derivative err:  mae={metrics['eq5_ce_derivative_mae_nats']:.6f}  "
              f"rmse={metrics['eq5_ce_derivative_rmse_nats']:.6f}")
        print(f"Outputs saved to:        {output_dir}")

    return {
        "run_id": run_id,
        "output_dir": output_dir,
        "metrics": metrics,
        "t_grid": t_grid,
        "model_curves": model_curves,
        "reference_curves": reference_curves,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = compose_config(PROJECT_ROOT, sys.argv[1:])
    config_name, overrides = resolve_config_name_and_overrides(sys.argv[1:])
    result = run_experiment(cfg, project_root=PROJECT_ROOT)

    metrics = result["metrics"]
    print(f"Reference entropy:       {metrics['reference_entropy_nats']:.6f} nats")
    print(f"Model CE area (eq. 6):   {metrics['model_ce_area_nats']:.6f}  "
          f"rel={metrics['model_ce_area_rel_error']:.2%}")
    print(f"Model eq. (5) dCE/dt:    mean={metrics['model_eq5_ce_derivative_mean_nats']:.6f}  "
          f"std={metrics['model_eq5_ce_derivative_std_nats']:.6f}")
    print(f"Eq. (5) derivative err:  mae={metrics['eq5_ce_derivative_mae_nats']:.6f}  "
          f"rmse={metrics['eq5_ce_derivative_rmse_nats']:.6f}")
    print(f"Saved outputs under:     {result['output_dir']}")


if __name__ == "__main__":
    main()
