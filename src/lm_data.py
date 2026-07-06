"""Language-model benchmark loading and token block packing."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import torch

from .lm_tokenizers import TokenizerMetadata, build_tokenizer


def _resolve_path(base_dir: Path, value: str | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _bytes_to_long_tensor(payload: bytes) -> torch.Tensor:
    view = memoryview(bytearray(payload))
    tensor = torch.frombuffer(view, dtype=torch.uint8).clone()
    return tensor.to(dtype=torch.long)


def _tokens_file_to_long_tensor(path: Path) -> torch.Tensor:
    payload = path.read_bytes()
    if len(payload) % 4 != 0:
        raise ValueError(f"Token file {path} has {len(payload)} bytes, which is not divisible by 4.")
    if not payload:
        return torch.empty(0, dtype=torch.long)
    view = memoryview(bytearray(payload))
    tensor = torch.frombuffer(view, dtype=torch.int32).clone()
    return tensor.to(dtype=torch.long)


def _token_file_count(path: Path) -> int:
    size_bytes = int(path.stat().st_size)
    if size_bytes % 4 != 0:
        raise ValueError(f"Token file {path} has {size_bytes} bytes, which is not divisible by 4.")
    return size_bytes // 4


@dataclass(frozen=True, slots=True)
class _TrainShard:
    path: Path
    token_count: int


@dataclass(slots=True)
class _ShardBackedTrainSource:
    shards: tuple[_TrainShard, ...]
    _loaded_shards: dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    _window_cache: dict[int, tuple[torch.Tensor, int]] = field(default_factory=dict, init=False, repr=False)
    _cache_order: list[int] = field(default_factory=list, init=False, repr=False)
    _max_cached_shards: int = field(default=2, init=False, repr=False)

    @property
    def token_count(self) -> int:
        return sum(int(shard.token_count) for shard in self.shards)

    def sample_batch(
        self,
        *,
        batch_size: int,
        window: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        boundaries, total_valid_starts = self._window_boundaries(window)
        if total_valid_starts <= 0:
            raise ValueError(
                f"Training split has no shard with at least {window} tokens required for the current context window."
            )
        starts = torch.randint(total_valid_starts, (int(batch_size),), generator=generator, dtype=torch.long)
        shard_indices = torch.searchsorted(boundaries, starts, right=True)
        offsets = torch.arange(window, dtype=torch.long)
        batch = torch.empty((int(batch_size), window), dtype=torch.long)
        for shard_idx in shard_indices.unique(sorted=True).tolist():
            shard_mask = shard_indices == int(shard_idx)
            shard_starts = starts[shard_mask]
            previous = 0 if shard_idx == 0 else int(boundaries[shard_idx - 1].item())
            local_starts = shard_starts - previous
            tokens = self._load_shard(int(shard_idx))
            batch[shard_mask] = tokens[local_starts.unsqueeze(1) + offsets.unsqueeze(0)]
        return batch

    def materialize(self) -> torch.Tensor:
        if not self.shards:
            return torch.empty(0, dtype=torch.long)
        return torch.cat([self._load_shard(idx) for idx in range(len(self.shards))], dim=0)

    def _load_shard(self, shard_idx: int) -> torch.Tensor:
        cached = self._loaded_shards.get(int(shard_idx))
        if cached is not None:
            if shard_idx in self._cache_order:
                self._cache_order.remove(int(shard_idx))
            self._cache_order.append(int(shard_idx))
            return cached
        tokens = _tokens_file_to_long_tensor(self.shards[int(shard_idx)].path)
        self._loaded_shards[int(shard_idx)] = tokens
        self._cache_order.append(int(shard_idx))
        while len(self._cache_order) > self._max_cached_shards:
            evict_idx = self._cache_order.pop(0)
            self._loaded_shards.pop(int(evict_idx), None)
        return tokens

    def _window_boundaries(self, window: int) -> tuple[torch.Tensor, int]:
        cached = self._window_cache.get(int(window))
        if cached is not None:
            return cached
        counts = [max(0, int(shard.token_count) - int(window) + 1) for shard in self.shards]
        boundaries = torch.tensor(counts, dtype=torch.long).cumsum(dim=0)
        total_valid_starts = int(boundaries[-1].item()) if len(counts) > 0 else 0
        payload = (boundaries, total_valid_starts)
        self._window_cache[int(window)] = payload
        return payload


@dataclass(slots=True)
class LMBenchmark:
    name: str
    data_dir: Path
    context_length: int
    tokenizer: TokenizerMetadata
    train_tokens: torch.Tensor | None
    val_tokens: torch.Tensor
    test_tokens: torch.Tensor
    train_source: _ShardBackedTrainSource | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _eval_cache: dict[tuple[str, bool], torch.Tensor] = field(default_factory=dict, init=False, repr=False)

    @property
    def vocab_size(self) -> int:
        return int(self.tokenizer.vocab_size)

    def split_tokens(self, split: str) -> torch.Tensor:
        split_name = str(split).lower()
        if split_name == "train":
            if self.train_tokens is None:
                assert self.train_source is not None
                return self.train_source.materialize()
            return self.train_tokens
        if split_name == "val":
            return self.val_tokens
        if split_name == "test":
            return self.test_tokens
        raise ValueError(f"Unsupported split {split!r}.")

    def sample_train_batch(
        self,
        *,
        batch_size: int,
        device: torch.device | str,
        include_next_token: bool,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        window = self.context_length + 1 if include_next_token else self.context_length
        if self.train_source is not None:
            batch = self.train_source.sample_batch(batch_size=int(batch_size), window=window, generator=generator)
        else:
            assert self.train_tokens is not None
            tokens = self.train_tokens
            if int(tokens.numel()) < window:
                raise ValueError(
                    f"Training split for {self.name} has only {int(tokens.numel())} tokens, "
                    f"but requires at least {window} for context_length={self.context_length}."
                )
            max_start = int(tokens.numel()) - window + 1
            starts = torch.randint(max_start, (int(batch_size),), generator=generator, dtype=torch.long)
            offsets = torch.arange(window, dtype=torch.long)
            batch = tokens[starts.unsqueeze(1) + offsets.unsqueeze(0)]
        return batch.to(device=device, dtype=torch.long, non_blocking=True)

    def iter_eval_batches(
        self,
        *,
        split: str,
        batch_size: int,
        device: torch.device | str,
        include_next_token: bool,
        max_batches: int | None = None,
        shard_rank: int = 0,
        num_shards: int = 1,
    ):
        blocks = self._eval_blocks(split=split, include_next_token=include_next_token)
        if max_batches is not None:
            blocks = blocks[: int(batch_size) * int(max_batches)]
        if int(num_shards) > 1:
            blocks = blocks[int(shard_rank) :: int(num_shards)]
        total_blocks = int(blocks.shape[0])
        if total_blocks <= 0:
            if int(num_shards) > 1:
                return
            raise ValueError(
                f"Split {split!r} for benchmark {self.name} does not contain a full evaluation block "
                f"for context_length={self.context_length}."
            )
        emitted = 0
        for start in range(0, total_blocks, int(batch_size)):
            batch = blocks[start : start + int(batch_size)]
            emitted += 1
            yield batch.to(device=device, dtype=torch.long, non_blocking=True)

    def token_counts(self) -> dict[str, int]:
        return {
            "train": int(self.train_source.token_count if self.train_source is not None else self.train_tokens.numel()),
            "val": int(self.val_tokens.numel()),
            "test": int(self.test_tokens.numel()),
        }

    def benchmark_metadata(self) -> dict[str, Any]:
        return {
            "benchmark_name": self.name,
            "data_dir": str(self.data_dir),
            "context_length": int(self.context_length),
            "tokenizer_name": self.tokenizer.name,
            "vocab_size": int(self.vocab_size),
            "token_unit": self.tokenizer.token_unit,
            "split_token_counts": self.token_counts(),
            **self.metadata,
        }

    def _eval_blocks(self, *, split: str, include_next_token: bool) -> torch.Tensor:
        cache_key = (str(split).lower(), bool(include_next_token))
        cached = self._eval_cache.get(cache_key)
        if cached is not None:
            return cached
        tokens = self.split_tokens(split)
        window = self.context_length + 1 if include_next_token else self.context_length
        n_blocks = int(tokens.numel()) // window
        if n_blocks <= 0:
            blocks = torch.empty((0, window), dtype=torch.long)
        else:
            trimmed = tokens[: n_blocks * window]
            blocks = trimmed.view(n_blocks, window).contiguous()
        self._eval_cache[cache_key] = blocks
        return blocks


def load_lm_benchmark(cfg, *, project_root: Path) -> LMBenchmark:
    benchmark_name = str(cfg.name).lower()
    base_dir = _resolve_path(project_root, str(cfg.data_dir))
    assert base_dir is not None
    tokenizer = build_tokenizer(str(cfg.tokenizer.name))
    tokenizer_meta = tokenizer.metadata()

    if benchmark_name == "text8":
        train_tokens, train_source, val_tokens, test_tokens, extra = _load_byte_corpus(
            cfg,
            base_dir=base_dir,
            benchmark_name=benchmark_name,
        )
    elif benchmark_name == "lm1b":
        train_tokens, train_source, val_tokens, test_tokens, extra = _load_lm1b(cfg, base_dir=base_dir)
    else:
        raise ValueError(f"Unsupported benchmark {cfg.name!r}.")

    return LMBenchmark(
        name=benchmark_name,
        data_dir=base_dir,
        context_length=int(cfg.context_length),
        tokenizer=tokenizer_meta,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        test_tokens=test_tokens,
        train_source=train_source,
        metadata=extra,
    )


def _bootstrap_hint(dataset_name: str) -> str:
    return (
        f"Run `python code/scripts/download_lm_datasets.py --dataset {dataset_name}` "
        "from the repo root to prepare the benchmark files."
    )


def _load_byte_corpus(
    cfg,
    *,
    base_dir: Path,
    benchmark_name: str,
) -> tuple[torch.Tensor, _ShardBackedTrainSource | None, torch.Tensor, torch.Tensor, dict[str, Any]]:
    raw_path = _resolve_path(base_dir, getattr(cfg.files, "raw", None))
    if raw_path is not None and raw_path.is_file():
        payload = raw_path.read_bytes()
        train_bytes = int(cfg.splits.train_bytes)
        val_bytes = int(cfg.splits.val_bytes)
        test_bytes = int(cfg.splits.test_bytes)
        expected = train_bytes + val_bytes + test_bytes
        if len(payload) < expected:
            raise ValueError(
                f"Raw {benchmark_name} file {raw_path} has {len(payload)} bytes but {expected} bytes are required "
                "for the configured train/val/test split."
            )
        train_tokens = _bytes_to_long_tensor(payload[:train_bytes])
        val_tokens = _bytes_to_long_tensor(payload[train_bytes : train_bytes + val_bytes])
        test_tokens = _bytes_to_long_tensor(payload[train_bytes + val_bytes : expected])
        metadata = {
            "source_format": f"raw_{benchmark_name}",
            "raw_path": str(raw_path),
            "split_bytes": {"train": train_bytes, "val": val_bytes, "test": test_bytes},
        }
        return train_tokens, None, val_tokens, test_tokens, metadata

    train_path = _resolve_path(base_dir, getattr(cfg.files, "train", None))
    val_path = _resolve_path(base_dir, getattr(cfg.files, "val", None))
    test_path = _resolve_path(base_dir, getattr(cfg.files, "test", None))
    missing = [str(path) for path in (train_path, val_path, test_path) if path is None or not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{benchmark_name} benchmark requires either a raw file or train/val/test byte files. Missing: "
            + ", ".join(missing)
            + ". "
            + _bootstrap_hint(benchmark_name)
        )
    return (
        _bytes_to_long_tensor(train_path.read_bytes()),
        None,
        _bytes_to_long_tensor(val_path.read_bytes()),
        _bytes_to_long_tensor(test_path.read_bytes()),
        {
            "source_format": "pre_split_bytes",
            "train_path": str(train_path),
            "val_path": str(val_path),
            "test_path": str(test_path),
        },
    )


def _load_lm1b(
    cfg,
    *,
    base_dir: Path,
) -> tuple[None, _ShardBackedTrainSource, torch.Tensor, torch.Tensor, dict[str, Any]]:
    train_dir = _resolve_path(base_dir, getattr(cfg.files, "train_dir", None))
    val_path = _resolve_path(base_dir, getattr(cfg.files, "val", None))
    test_path = _resolve_path(base_dir, getattr(cfg.files, "test", None))
    dataset_info_path = base_dir / "dataset_info.json"

    missing: list[str] = []
    if train_dir is None or not train_dir.is_dir():
        missing.append(str(train_dir))
    if val_path is None or not val_path.is_file():
        missing.append(str(val_path))
    if test_path is None or not test_path.is_file():
        missing.append(str(test_path))
    if missing:
        raise FileNotFoundError(
            "LM1B benchmark files are missing: " + ", ".join(missing) + ". " + _bootstrap_hint("lm1b")
        )

    shard_paths = sorted(path for path in train_dir.iterdir() if path.is_file() and path.suffix == ".tokens")
    if not shard_paths:
        raise FileNotFoundError(f"LM1B training shard directory is empty: {train_dir}. " + _bootstrap_hint("lm1b"))

    shards = tuple(_TrainShard(path=path, token_count=_token_file_count(path)) for path in shard_paths)
    if not any(int(shard.token_count) >= int(cfg.context_length) for shard in shards):
        raise ValueError(
            f"LM1B training shards under {train_dir} are all shorter than context_length={int(cfg.context_length)}."
        )

    metadata: dict[str, Any] = {
        "source_format": "int32_token_shards",
        "train_dir": str(train_dir),
        "train_shard_count": len(shards),
        "train_shard_paths": [str(path) for path in shard_paths],
        "val_path": str(val_path),
        "test_path": str(test_path),
    }
    if dataset_info_path.is_file():
        dataset_info = json.loads(dataset_info_path.read_text(encoding="utf-8"))
        for key in (
            "hf_dataset",
            "hf_config",
            "hf_split_map",
            "validation_rows",
            "train_shard_target_tokens",
            "source_format",
        ):
            if key in dataset_info:
                metadata[key] = dataset_info[key]

    return (
        None,
        _ShardBackedTrainSource(shards=shards),
        _tokens_file_to_long_tensor(val_path),
        _tokens_file_to_long_tensor(test_path),
        metadata,
    )
