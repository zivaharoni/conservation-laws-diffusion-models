#!/usr/bin/env python3
"""Thin local MC-vs-integration check for LM conservation metrics."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from src.lm_data import load_lm_benchmark  # noqa: E402
from src.lm_models import ContinuousDiffusionTransformerLM, MaskedTransformerLM  # noqa: E402
from src.lm_objectives import (  # noqa: E402
    evaluate_exit_curve,
    evaluate_exit_objective_metrics,
    evaluate_immse_curve,
    evaluate_immse_objective_metrics,
    evaluate_qsc_curve,
    evaluate_qsc_objective_metrics,
)


def _cfg(context_length: int) -> SimpleNamespace:
    return SimpleNamespace(
        name="text8",
        data_dir="data/lm/text8",
        context_length=int(context_length),
        tokenizer=SimpleNamespace(name="byte"),
        files=SimpleNamespace(raw="text8", train="train.bin", val="val.bin", test="test.bin"),
        splits=SimpleNamespace(train_bytes=90_000_000, val_bytes=5_000_000, test_bytes=5_000_000),
    )


def _masked_model(benchmark, hidden_size: int) -> MaskedTransformerLM:
    return MaskedTransformerLM(
        vocab_size=benchmark.vocab_size,
        hidden_size=int(hidden_size),
        n_layers=2,
        n_heads=4,
        ffn_hidden_size=int(hidden_size) * 2,
        max_seq_len=benchmark.context_length,
        dropout=0.0,
        use_timestep_conditioning=True,
    )


def _immse_model(benchmark, hidden_size: int) -> ContinuousDiffusionTransformerLM:
    return ContinuousDiffusionTransformerLM(
        vocab_size=benchmark.vocab_size,
        hidden_size=int(hidden_size),
        n_layers=2,
        n_heads=4,
        ffn_hidden_size=int(hidden_size) * 2,
        max_seq_len=benchmark.context_length,
        dropout=0.0,
    )


def _row(name: str, mc: float, integral: float, **extra: float | int) -> dict[str, float | int | str]:
    abs_diff = abs(float(mc) - float(integral))
    denom = max(abs(float(integral)), 1.0e-12)
    return {
        "objective": name,
        "mc_conservation_law_nats": float(mc),
        "integral_conservation_law_nats": float(integral),
        "abs_diff_nats": float(abs_diff),
        "rel_diff": float(abs_diff / denom),
        **extra,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--context-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=2)
    parser.add_argument("--mc-steps", type=int, default=24)
    parser.add_argument("--curve-steps", type=int, default=8)
    parser.add_argument("--points", type=int, default=9)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--objectives", default="exit,qsc,immse")
    parser.add_argument("--qsc-t-min", type=float, default=0.1)
    parser.add_argument("--qsc-t-max", type=float, default=0.9)
    parser.add_argument("--snr-min", type=float, default=1.0e-3)
    parser.add_argument("--snr-max", type=float, default=1.0)
    args = parser.parse_args()

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    torch.set_num_threads(1)

    benchmark = load_lm_benchmark(_cfg(args.context_length), project_root=PROJECT_ROOT)
    metadata = benchmark.benchmark_metadata()
    print("benchmark:", json.dumps(metadata, sort_keys=True))
    print(
        "settings:",
        json.dumps(
            {
                "split": args.split,
                "batch_size": args.batch_size,
                "max_batches": args.max_batches,
                "mc_steps": args.mc_steps,
                "curve_steps": args.curve_steps,
                "points": args.points,
                "seed": args.seed,
            },
            sort_keys=True,
        ),
    )

    rows: list[dict[str, float | int | str]] = []
    objectives = {item.strip().lower() for item in str(args.objectives).split(",") if item.strip()}

    if "exit" in objectives:
        exit_model = _masked_model(benchmark, args.hidden_size)
        exit_mc = evaluate_exit_objective_metrics(
            exit_model,
            benchmark,
            split=args.split,
            batch_size=args.batch_size,
            device="cpu",
            min_rate=0.0,
            max_rate=1.0,
            max_batches=args.max_batches,
            steps_per_batch=args.mc_steps,
            seed=args.seed + 11,
        )
        _s_grid, _exit_curve, exit_integral, _exit_rows = evaluate_exit_curve(
            exit_model,
            benchmark,
            split=args.split,
            batch_size=args.batch_size,
            device="cpu",
            n_s_points=args.points,
            max_batches=args.max_batches,
            steps_per_batch=args.curve_steps,
            seed=args.seed + 12,
        )
        rows.append(
            _row(
                "exit",
                exit_mc["conservation_law_nats"],
                exit_integral["curve_integral_nats"],
                evaluated_tokens=exit_mc["evaluated_tokens"],
            )
        )

    if "qsc" in objectives:
        qsc_model = _masked_model(benchmark, args.hidden_size)
        qsc_t_min = 1.0 / float(benchmark.vocab_size) if args.qsc_t_min is None else float(args.qsc_t_min)
        qsc_t_max = float(args.qsc_t_max)
        qsc_mc = evaluate_qsc_objective_metrics(
            qsc_model,
            benchmark,
            split=args.split,
            batch_size=args.batch_size,
            device="cpu",
            t_min=qsc_t_min,
            t_max=qsc_t_max,
            max_batches=args.max_batches,
            steps_per_batch=args.mc_steps,
            seed=args.seed + 21,
        )
        _t_grid, _qsc_curve, qsc_integral, _qsc_rows = evaluate_qsc_curve(
            qsc_model,
            benchmark,
            split=args.split,
            batch_size=args.batch_size,
            device="cpu",
            t_min=qsc_t_min,
            t_max=qsc_t_max,
            n_t_points=args.points,
            max_batches=args.max_batches,
            steps_per_batch=args.curve_steps,
            seed=args.seed + 22,
        )
        rows.append(
            _row(
                "qsc",
                qsc_mc["conservation_law_nats"],
                qsc_integral["curve_integral_nats"],
                evaluated_tokens=qsc_mc["evaluated_tokens"],
            )
        )

    if "immse" in objectives:
        immse_model = _immse_model(benchmark, args.hidden_size)
        snr_min = float(args.snr_min)
        snr_max = float(args.snr_max)
        immse_mc = evaluate_immse_objective_metrics(
            immse_model,
            benchmark,
            split=args.split,
            batch_size=args.batch_size,
            device="cpu",
            snr_min=snr_min,
            snr_max=snr_max,
            max_batches=args.max_batches,
            steps_per_batch=args.mc_steps,
            seed=args.seed + 31,
        )
        _snr_grid, _mmse_curve, immse_integral, _immse_rows = evaluate_immse_curve(
            immse_model,
            benchmark,
            split=args.split,
            batch_size=args.batch_size,
            device="cpu",
            snr_min=snr_min,
            snr_max=snr_max,
            n_snr_points=args.points,
            max_batches=args.max_batches,
            steps_per_batch=args.curve_steps,
            seed=args.seed + 32,
        )
        rows.append(
            _row(
                "immse",
                immse_mc["conservation_law_nats"],
                immse_integral["curve_integral_nats"],
                evaluated_tokens=immse_mc["evaluated_tokens"],
            )
        )

    print("results:")
    for row in rows:
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
