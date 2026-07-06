from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common.config import compose_config
from experiments.common.io import ensure_dir, save_config, save_csv, save_json
from experiments.common.repro import configure_environment, get_best_device, make_run_id, set_global_seed
from experiments.common.wandb_logger import WandbLogger
from src.models import CausalCrossEntropySequenceModel
from src.sources_markov import (
    build_markov_chain,
    sample_markov_sequences_torch,
)


def _count_parameters(model: nn.Module) -> int:
    return sum(int(param.numel()) for param in model.parameters())


@dataclass(slots=True)
class BudgetSpec:
    alphabet_size: int
    batch_size: int
    seed: int
    device: str
    n_steps: int
    lr: float
    grad_clip: float
    seq_len: int
    eval_batches: int
    log_every: int
    target_params: int
    model_kwargs: dict[str, object]


@dataclass(slots=True)
class FamilySampler:
    family: str
    metadata: dict[str, object]
    reference_entropy_nats: float
    _train_sampler: Callable[[int, int, torch.device | str], torch.Tensor]
    _eval_sampler: Callable[[int, int, torch.device | str], torch.Tensor]

    def sample_train(self, batch_size: int, seq_len: int, device: torch.device | str) -> torch.Tensor:
        return self._train_sampler(batch_size, seq_len, device)

    def sample_eval(self, batch_size: int, seq_len: int, device: torch.device | str) -> torch.Tensor:
        return self._eval_sampler(batch_size, seq_len, device)


DEFAULT_CONFIG_NAME = "experiments/markov_scaling/causal_ce"


def _extract_config_name(argv: list[str]) -> tuple[str, list[str], list[str]]:
    config_name = DEFAULT_CONFIG_NAME
    hydra_overrides: list[str] = []
    remaining: list[str] = []
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg == "--config-name":
            if idx + 1 >= len(argv):
                raise ValueError("--config-name requires a value")
            config_name = argv[idx + 1]
            idx += 2
            continue
        if arg.startswith("config_name="):
            config_name = arg.split("=", 1)[1]
            idx += 1
            continue
        if "=" in arg and not arg.startswith("--"):
            hydra_overrides.append(arg)
            idx += 1
            continue
        remaining.append(arg)
        idx += 1
    return config_name, hydra_overrides, remaining


def _build_run_cfg(
    *,
    args: argparse.Namespace,
    budget: BudgetSpec,
    sampler: FamilySampler,
    config_name: str,
    hydra_overrides: list[str] | None = None,
    project_root: Path = PROJECT_ROOT,
) -> object:
    base_cfg = compose_config(project_root=project_root, argv=[f"config_name={config_name}", *(hydra_overrides or [])])
    cfg = OmegaConf.create(
        {
            "name": f"causal_ce_{sampler.family}",
            "logging": OmegaConf.to_container(base_cfg.logging, resolve=True),
            "paths": {"wandb_dir_name": "wandb"},
            "family": sampler.family,
            "data": sampler.metadata,
            "model": budget.model_kwargs,
            "train": {
                "n_steps": int(args.n_steps),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "grad_clip": float(args.grad_clip),
                "seq_len": int(args.seq_len),
            },
            "eval": {
                "eval_batches": int(args.eval_batches),
            },
            "seed": int(args.seed),
            "results_root": str(args.results_root),
        }
    )
    cfg.logging.tags = ["publish", "causal_ce", sampler.family]
    cfg.logging.group = f"causal_ce_{sampler.family}"
    return cfg


def _load_budget_spec(
    *,
    alphabet_size: int | None,
    seq_len: int | None,
    n_transformer_layers: int | None = None,
    config_name: str = DEFAULT_CONFIG_NAME,
    hydra_overrides: list[str] | None = None,
    project_root: Path = PROJECT_ROOT,
) -> BudgetSpec:
    overrides: list[str] = [f"config_name={config_name}", *(hydra_overrides or [])]
    if alphabet_size is not None:
        overrides.append(f"data.alphabet_size={int(alphabet_size)}")
    if seq_len is not None:
        overrides.append(f"train.seq_len={int(seq_len)}")
        overrides.append(f"eval.seq_len={int(seq_len)}")
    if n_transformer_layers is not None:
        overrides.append(f"model.n_transformer_layers={int(n_transformer_layers)}")

    cfg = compose_config(project_root=project_root, argv=overrides)

    grad_clip_cfg = cfg.train.grad_clip
    grad_clip_value = max(
        float(grad_clip_cfg.embedding),
        float(grad_clip_cfg.attention),
        float(grad_clip_cfg.alphabet_proj),
        float(grad_clip_cfg.other),
    )
    log_every = max(1, int(getattr(cfg.logging, "log_every", 100)))

    budget = BudgetSpec(
        alphabet_size=int(cfg.data.alphabet_size),
        batch_size=int(cfg.train.batch_size),
        seed=int(cfg.seed),
        device=str(cfg.device),
        n_steps=int(cfg.train.n_steps),
        lr=float(cfg.train.lr),
        grad_clip=grad_clip_value,
        seq_len=int(cfg.train.seq_len),
        eval_batches=int(cfg.eval.eval_batches),
        log_every=log_every,
        target_params=0,
        model_kwargs={
            "alphabet_size": int(cfg.data.alphabet_size),
            "embed_dim": int(cfg.model.embed_dim),
            "n_transformer_layers": int(cfg.model.n_transformer_layers),
            "n_heads": int(getattr(cfg.model, "n_heads", 4)),
            "attention_dropout": float(getattr(cfg.model, "attention_dropout", 0.0)),
            "max_seq_len": max(int(cfg.train.seq_len), int(cfg.eval.seq_len)),
        },
    )
    target_model = CausalCrossEntropySequenceModel(**budget.model_kwargs)
    budget.target_params = _count_parameters(target_model)
    return budget


def _make_markov_sampler(
    *,
    alphabet_size: int,
    dirichlet_alpha: float,
    seed: int,
    init_distribution: str,
) -> FamilySampler:
    base_chain = build_markov_chain(
        alphabet_size=alphabet_size,
        dirichlet_alpha=dirichlet_alpha,
        seed=seed,
        init_distribution=init_distribution,
    )
    chain = base_chain

    init_probs = torch.as_tensor(chain.init_probs, dtype=torch.float32, device="cpu")
    transition_matrix = torch.as_tensor(chain.transition_matrix, dtype=torch.float32, device="cpu")
    train_generator = torch.Generator(device="cpu")
    train_generator.manual_seed(int(seed) + 201)
    eval_generator = torch.Generator(device="cpu")
    eval_generator.manual_seed(int(seed) + 202)

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
        return states.to(device=device, dtype=torch.long)

    metadata = {
        "family": "markov",
        "alphabet_size": int(alphabet_size),
        "dirichlet_alpha": float(dirichlet_alpha),
        "init_distribution": str(init_distribution),
    }

    return FamilySampler(
        family="markov",
        metadata=metadata,
        reference_entropy_nats=float(base_chain.entropy_rate_nats),
        _train_sampler=lambda batch_size, seq_len, device: _sample_with_generator(
            train_generator, batch_size, seq_len, device
        ),
        _eval_sampler=lambda batch_size, seq_len, device: _sample_with_generator(
            eval_generator, batch_size, seq_len, device
        ),
    )


def _build_family_sampler(args: argparse.Namespace, family: str) -> FamilySampler:
    if family != "markov":
        raise ValueError(f"Unsupported family={family!r}; expected 'markov'.")
    return _make_markov_sampler(
        alphabet_size=int(args.alphabet_size),
        dirichlet_alpha=float(args.dirichlet_alpha),
        seed=int(args.seed),
        init_distribution=str(args.init_distribution),
    )


@torch.no_grad()
def _evaluate_cross_entropy(
    model: nn.Module,
    sampler: FamilySampler,
    *,
    batch_size: int,
    seq_len: int,
    eval_batches: int,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for _ in range(eval_batches):
        tokens = sampler.sample_eval(batch_size=batch_size, seq_len=seq_len, device=device)
        logits = model(tokens[:, :-1])
        targets = tokens[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum")
        total_loss += float(loss.item())
        total_tokens += int(targets.numel())
    return total_loss / max(total_tokens, 1)


def _train_family(
    *,
    args: argparse.Namespace,
    budget: BudgetSpec,
    family: str,
    device: torch.device,
    results_root: Path,
    config_name: str,
) -> dict[str, object]:
    sampler = _build_family_sampler(args, family)
    run_id = make_run_id(
        f"causal_ce_{family}",
        [
            f"k={args.alphabet_size}",
            f"seq={args.seq_len}",
            f"layers={budget.model_kwargs['n_transformer_layers']}",
            f"steps={args.n_steps}",
            f"seed={args.seed}",
        ],
    )
    output_dir = ensure_dir(results_root / family / run_id)
    hydra_overrides = list(getattr(args, "hydra_overrides", []))
    run_cfg = _build_run_cfg(
        args=args,
        budget=budget,
        sampler=sampler,
        config_name=config_name,
        hydra_overrides=hydra_overrides,
    )
    save_config(output_dir / "config.yaml", run_cfg)
    logger = WandbLogger(cfg=run_cfg, run_id=run_id, output_dir=str(output_dir / run_cfg.paths.wandb_dir_name))

    model = CausalCrossEntropySequenceModel(**budget.model_kwargs).to(device)
    causal_params = _count_parameters(model)
    if causal_params != budget.target_params:
        raise RuntimeError(
            f"Causal CE model budget mismatch for {family}: "
            f"causal_params={causal_params:,} target_params={budget.target_params:,}."
        )

    logger.log(
        {
            "model/parameter_count": int(causal_params),
            "reference/entropy_nats_per_token": float(sampler.reference_entropy_nats),
            "reference/entropy_bits_per_token": float(sampler.reference_entropy_nats / math.log(2.0)),
        }
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    history_rows: list[dict[str, float | int]] = []

    for step in range(1, int(args.n_steps) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        tokens = sampler.sample_train(batch_size=int(args.batch_size), seq_len=int(args.seq_len), device=device)
        logits = model(tokens[:, :-1])
        targets = tokens[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="mean")
        loss.backward()

        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip)).item()
        )
        optimizer.step()

        train_ce_nats = float(loss.item())
        train_ce_bits = train_ce_nats / math.log(2.0)
        row = {
            "step": int(step),
            "train_ce_nats": train_ce_nats,
            "train_ce_bits": train_ce_bits,
            "grad_norm": grad_norm,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history_rows.append(row)

        if step == 1 or step % int(args.log_every) == 0 or step == int(args.n_steps):
            reference_nats = float(sampler.reference_entropy_nats)
            logger.log(
                {
                    "train/ce_nats": train_ce_nats,
                    "train/ce_bits": train_ce_bits,
                    "train/reference_entropy_nats": reference_nats,
                    "train/excess_ce_nats": train_ce_nats - reference_nats,
                    "train/grad_norm": grad_norm,
                    "train/lr": float(optimizer.param_groups[0]["lr"]),
                },
                step=step,
            )
            print(
                f"[{family}] step={step} train_ce_nats={train_ce_nats:.6f} "
                f"reference_nats={reference_nats:.6f} "
                f"grad_norm={grad_norm:.6f}"
            )

    eval_ce_nats = _evaluate_cross_entropy(
        model,
        sampler,
        batch_size=int(args.batch_size),
        seq_len=int(args.seq_len),
        eval_batches=int(args.eval_batches),
        device=device,
    )
    eval_ce_bits = eval_ce_nats / math.log(2.0)
    reference_nats = float(sampler.reference_entropy_nats)
    reference_bits = reference_nats / math.log(2.0)
    excess_nats = eval_ce_nats - reference_nats
    rel_gap = excess_nats / max(abs(reference_nats), 1e-12)

    metrics = {
        "family": family,
        "device": str(device),
        "run_id": run_id,
        "seed": int(args.seed),
        "alphabet_size": int(args.alphabet_size),
        "seq_len": int(args.seq_len),
        "n_steps": int(args.n_steps),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "grad_clip": float(args.grad_clip),
        "target_params": int(budget.target_params),
        "causal_model_params": int(causal_params),
        "eval_batches": int(args.eval_batches),
        "eval_ce_nats_per_token": float(eval_ce_nats),
        "eval_ce_bits_per_token": float(eval_ce_bits),
        "reference_entropy_nats_per_token": float(reference_nats),
        "reference_entropy_bits_per_token": float(reference_bits),
        "excess_ce_nats_per_token": float(excess_nats),
        "relative_gap": float(rel_gap),
        **sampler.metadata,
    }

    save_csv(output_dir / "history.csv", history_rows)
    save_json(output_dir / "metrics.json", metrics)
    torch.save(model.state_dict(), output_dir / "model.pt")
    logger.log({"eval": metrics})
    logger.log_directory_files(
        output_dir,
        artifact_name=f"{run_id}-outputs",
        exclude_dirs=[str(run_cfg.paths.wandb_dir_name)],
    )
    print(
        f"[{family}] done eval_ce_nats={eval_ce_nats:.6f} reference_nats={reference_nats:.6f} "
        f"gap_nats={excess_nats:.6f} saved={output_dir}"
    )
    return {
        "metrics": metrics,
        "logger": logger,
        "run_id": run_id,
        "output_dir": output_dir,
    }


def _build_parser(defaults: BudgetSpec) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the causal cross-entropy baseline for Markov data."
    )
    parser.add_argument(
        "--family",
        type=str,
        default="markov",
        choices=["markov"],
        help="Source family to train.",
    )
    parser.add_argument("--alphabet-size", type=int, default=int(defaults.alphabet_size))
    parser.add_argument("--seq-len", type=int, default=int(defaults.seq_len))
    parser.add_argument("--n-steps", type=int, default=int(defaults.n_steps))
    parser.add_argument("--batch-size", type=int, default=int(defaults.batch_size))
    parser.add_argument("--lr", type=float, default=float(defaults.lr))
    parser.add_argument("--grad-clip", type=float, default=float(defaults.grad_clip))
    parser.add_argument(
        "--n-transformer-layers",
        type=int,
        default=None,
        help="Override the standalone CE model depth. When set, the CE budget is recomputed for that depth.",
    )
    parser.add_argument("--seed", type=int, default=int(defaults.seed))
    parser.add_argument("--device", type=str, default=str(defaults.device))
    parser.add_argument("--eval-batches", type=int, default=int(defaults.eval_batches))
    parser.add_argument("--log-every", type=int, default=int(defaults.log_every))
    parser.add_argument("--dirichlet-alpha", type=float, default=0.05)
    parser.add_argument("--init-distribution", choices=["stationary", "sampled"], default="stationary")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "causal_ce",
        help="Directory where per-family outputs are written.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_environment()
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    config_name, hydra_overrides, parser_argv = _extract_config_name(raw_argv)
    initial_budget = _load_budget_spec(
        alphabet_size=None,
        seq_len=None,
        config_name=config_name,
        hydra_overrides=hydra_overrides,
    )
    parser = _build_parser(initial_budget)
    args = parser.parse_args(parser_argv)
    args.hydra_overrides = hydra_overrides

    if int(args.seq_len) < 2:
        raise ValueError(f"--seq-len must be at least 2 for next-token CE training, got {args.seq_len}.")
    if int(args.n_steps) <= 0:
        raise ValueError(f"--n-steps must be positive, got {args.n_steps}.")
    if int(args.batch_size) <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}.")
    if int(args.eval_batches) <= 0:
        raise ValueError(f"--eval-batches must be positive, got {args.eval_batches}.")
    if float(args.grad_clip) <= 0.0:
        raise ValueError(f"--grad-clip must be positive, got {args.grad_clip}.")
    if float(args.lr) <= 0.0:
        raise ValueError(f"--lr must be positive, got {args.lr}.")
    if args.n_transformer_layers is not None and int(args.n_transformer_layers) <= 0:
        raise ValueError(
            f"--n-transformer-layers must be positive when set, got {args.n_transformer_layers}."
        )

    budget = _load_budget_spec(
        alphabet_size=int(args.alphabet_size),
        seq_len=int(args.seq_len),
        n_transformer_layers=args.n_transformer_layers,
        config_name=config_name,
        hydra_overrides=hydra_overrides,
    )
    args.alphabet_size = int(budget.alphabet_size)
    args.seq_len = int(budget.seq_len)

    set_global_seed(int(args.seed), deterministic=False)
    device = get_best_device(str(args.device))
    results_root = ensure_dir(Path(args.results_root))

    print(
        f"[setup] target_markov_budget_params={budget.target_params:,} "
        f"alphabet_size={args.alphabet_size} seq_len={args.seq_len} "
        f"embed_dim={budget.model_kwargs['embed_dim']} "
        f"n_transformer_layers={budget.model_kwargs['n_transformer_layers']}"
    )

    family = str(args.family)
    results = [
        _train_family(
            args=args,
            budget=budget,
            family=family,
            device=device,
            results_root=results_root,
            config_name=config_name,
        )
    ]
    summary_rows = [
        {
            "family": item["metrics"]["family"],
            "eval_ce_bits_per_token": item["metrics"]["eval_ce_bits_per_token"],
            "reference_entropy_bits_per_token": item["metrics"]["reference_entropy_bits_per_token"],
            "relative_gap": item["metrics"]["relative_gap"],
            "run_id": item["metrics"]["run_id"],
        }
        for item in results
    ]
    summary_path = results_root / "summary.csv"
    save_csv(summary_path, summary_rows)
    for item in results:
        logger = item["logger"]
        logger.log_files([summary_path], artifact_name=f"{item['run_id']}-summary")
        logger.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
