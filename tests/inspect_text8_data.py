#!/usr/bin/env python3
"""Print text8 sequences through the LM benchmark loader for data review."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from src.lm_data import load_lm_benchmark  # noqa: E402


def _cfg(context_length: int) -> SimpleNamespace:
    return SimpleNamespace(
        name="text8",
        data_dir="data/lm/text8",
        context_length=int(context_length),
        tokenizer=SimpleNamespace(name="byte"),
        files=SimpleNamespace(raw="text8", train="train.bin", val="val.bin", test="test.bin"),
        splits=SimpleNamespace(train_bytes=90_000_000, val_bytes=5_000_000, test_bytes=5_000_000),
    )


def _decode(tokens: torch.Tensor) -> str:
    return bytes(int(token) % 256 for token in tokens.tolist()).decode("utf-8", errors="replace")


def _print_sequence(*, split: str, index: int, start: int, tokens: torch.Tensor, token_prefix: int) -> None:
    decoded = _decode(tokens)
    prefix = " ".join(str(int(token)) for token in tokens[:token_prefix].tolist())
    print(f"\n[{split} #{index}] start={start} length={int(tokens.numel())}")
    print(f"text: {decoded!r}")
    print(f"token_prefix[{min(token_prefix, int(tokens.numel()))}]: {prefix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect local text8 sequences loaded by src.lm_data.")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--num-sequences", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--token-prefix", type=int, default=32)
    parser.add_argument(
        "--actual-train-batch",
        action="store_true",
        help="Print rows returned by LMBenchmark.sample_train_batch, matching the training data path.",
    )
    parser.add_argument(
        "--include-next-token",
        action="store_true",
        help="Use the causal-CE training window of context_length + 1 tokens.",
    )
    parser.add_argument(
        "--random-eval",
        action="store_true",
        help="Sample validation/test starts randomly instead of printing contiguous blocks from offset 0.",
    )
    args = parser.parse_args()

    benchmark = load_lm_benchmark(_cfg(args.context_length), project_root=PROJECT_ROOT)
    metadata = benchmark.benchmark_metadata()
    print("benchmark:", metadata["benchmark_name"])
    print("data_dir:", metadata["data_dir"])
    print("context_length:", metadata["context_length"])
    print("tokenizer:", metadata["tokenizer_name"], f"({metadata['token_unit']})")
    print("token_counts:", metadata["split_token_counts"])
    print("source_format:", metadata["source_format"])

    generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
    if args.actual_train_batch:
        batch = benchmark.sample_train_batch(
            batch_size=int(args.num_sequences),
            device="cpu",
            include_next_token=bool(args.include_next_token),
            generator=generator,
        )
        mode = "causal_ce" if args.include_next_token else "denoising"
        print(f"actual_train_batch_mode: {mode}")
        for index, row in enumerate(batch, start=1):
            _print_sequence(
                split="train_batch",
                index=index,
                start=-1,
                tokens=row,
                token_prefix=int(args.token_prefix),
            )
        return

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    window = int(args.context_length)

    for split in splits:
        tokens = benchmark.split_tokens(split)
        max_start = int(tokens.numel()) - window + 1
        if max_start <= 0:
            print(f"\n[{split}] skipped: only {int(tokens.numel())} tokens for window={window}")
            continue

        if split == "train" or args.random_eval:
            starts = torch.randint(max_start, (int(args.num_sequences),), generator=generator, dtype=torch.long)
        else:
            starts = torch.arange(int(args.num_sequences), dtype=torch.long) * window
            starts = starts[starts < max_start]

        for index, start_t in enumerate(starts.tolist(), start=1):
            start = int(start_t)
            _print_sequence(
                split=split,
                index=index,
                start=start,
                tokens=tokens[start : start + window],
                token_prefix=int(args.token_prefix),
            )


if __name__ == "__main__":
    main()
