"""CIFAR-10 discrete image-token data utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch


def _resolve_path(base_dir: Path, value: str | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _load_tensor(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    tensor = torch.as_tensor(payload, dtype=torch.long).contiguous()
    if tensor.ndim != 2:
        raise ValueError(f"Expected token tensor [N, S] at {path}, got {tuple(tensor.shape)}")
    return tensor


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def extract_nonoverlapping_patches(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    if images.ndim != 4:
        raise ValueError(f"Expected images [N, H, W, C], got {tuple(images.shape)}")
    n_images, height, width, channels = images.shape
    patch_size = int(patch_size)
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(f"Image size {(height, width)} must be divisible by patch_size={patch_size}")
    grid_h = height // patch_size
    grid_w = width // patch_size
    patches = images.view(n_images, grid_h, patch_size, grid_w, patch_size, channels)
    patches = patches.permute(0, 1, 3, 2, 4, 5).contiguous()
    return patches.view(n_images, grid_h * grid_w, patch_size * patch_size * channels)


def reconstruct_images_from_patches(
    patch_tokens: torch.Tensor,
    *,
    codebook: torch.Tensor,
    patch_size: int,
    image_size: int = 32,
    mask_token_id: int | None = None,
    mask_fill_value: float = 0.5,
) -> torch.Tensor:
    tokens = torch.as_tensor(patch_tokens, dtype=torch.long, device="cpu")
    if tokens.ndim != 2:
        raise ValueError(f"Expected patch_tokens [N, S], got {tuple(tokens.shape)}")
    centers = torch.as_tensor(codebook, dtype=torch.float32, device="cpu")
    patch_dim = int(patch_size) * int(patch_size) * 3
    if centers.ndim != 2 or centers.shape[1] != patch_dim:
        raise ValueError(f"Expected codebook shape [K, {patch_dim}], got {tuple(centers.shape)}")
    grid_size = image_size // int(patch_size)
    seq_len = grid_size * grid_size
    if tokens.shape[1] != seq_len:
        raise ValueError(f"Expected sequence length {seq_len}, got {tokens.shape[1]}")
    safe_tokens = tokens.clone()
    mask = None
    if mask_token_id is not None:
        mask = safe_tokens == int(mask_token_id)
        safe_tokens = safe_tokens.clamp(min=0, max=int(centers.shape[0]) - 1)
    patches = centers[safe_tokens]
    if mask is not None and bool(mask.any()):
        patches[mask] = float(mask_fill_value)
    patches = patches.view(tokens.shape[0], grid_size, grid_size, int(patch_size), int(patch_size), 3)
    images = patches.permute(0, 1, 3, 2, 4, 5).contiguous()
    return images.view(tokens.shape[0], image_size, image_size, 3).clamp(0.0, 1.0)


def encode_rgb8_images(images: torch.Tensor) -> torch.Tensor:
    images_t = torch.as_tensor(images, device="cpu")
    if images_t.ndim != 4 or int(images_t.shape[-1]) != 3:
        raise ValueError(f"Expected images [N, H, W, 3], got {tuple(images_t.shape)}")
    if images_t.dtype.is_floating_point:
        images_t = torch.clamp(torch.round(images_t * 255.0), min=0.0, max=255.0).to(dtype=torch.long)
    else:
        images_t = images_t.to(dtype=torch.long)
    return images_t.view(int(images_t.shape[0]), -1).contiguous()


def reconstruct_images_from_rgb8_tokens(
    tokens: torch.Tensor,
    *,
    image_size: int = 32,
    mask_token_id: int | None = None,
    mask_fill_value: float = 0.5,
) -> torch.Tensor:
    tokens_t = torch.as_tensor(tokens, dtype=torch.long, device="cpu")
    expected_seq_len = int(image_size) * int(image_size) * 3
    if tokens_t.ndim != 2:
        raise ValueError(f"Expected tokens [N, S], got {tuple(tokens_t.shape)}")
    if int(tokens_t.shape[1]) != expected_seq_len:
        raise ValueError(f"Expected sequence length {expected_seq_len}, got {int(tokens_t.shape[1])}")
    safe_tokens = tokens_t.clone()
    mask = None
    if mask_token_id is not None:
        mask = safe_tokens == int(mask_token_id)
        safe_tokens = safe_tokens.clamp(min=0, max=255)
    images = safe_tokens.view(int(tokens_t.shape[0]), int(image_size), int(image_size), 3).to(dtype=torch.float32) / 255.0
    if mask is not None and bool(mask.any()):
        images[mask.view(int(tokens_t.shape[0]), int(image_size), int(image_size), 3)] = float(mask_fill_value)
    return images.clamp(0.0, 1.0)


def save_image_grid(
    output_path: Path,
    *,
    rows: list[tuple[str, torch.Tensor]],
    max_images: int = 4,
) -> None:
    if not rows:
        raise ValueError("rows must not be empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_count = min(int(max_images), min(int(images.shape[0]) for _, images in rows))
    fig, axes = plt.subplots(len(rows), image_count, figsize=(2.2 * image_count, 2.2 * len(rows)), squeeze=False)
    for row_idx, (label, images) in enumerate(rows):
        images_t = torch.as_tensor(images, dtype=torch.float32, device="cpu")
        for col_idx in range(image_count):
            ax = axes[row_idx][col_idx]
            ax.imshow(images_t[col_idx].clamp(0.0, 1.0).numpy())
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def fit_patch_codebook(
    patches: torch.Tensor,
    *,
    codebook_size: int,
    n_iters: int,
    batch_size: int,
    seed: int,
) -> torch.Tensor:
    samples = torch.as_tensor(patches, dtype=torch.float32, device="cpu")
    if samples.ndim != 2:
        raise ValueError(f"Expected patches [N, D], got {tuple(samples.shape)}")
    n_samples = int(samples.shape[0])
    if n_samples < int(codebook_size):
        raise ValueError(f"Need at least {codebook_size} patches, got {n_samples}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    init_idx = torch.randperm(n_samples, generator=generator)[: int(codebook_size)]
    centers = samples[init_idx].clone()
    chunk_size = max(1, int(batch_size))
    for _ in range(max(1, int(n_iters))):
        sums = torch.zeros_like(centers)
        counts = torch.zeros((int(codebook_size),), dtype=torch.float32)
        for start in range(0, n_samples, chunk_size):
            batch = samples[start : start + chunk_size]
            distances = torch.cdist(batch, centers)
            assignment = torch.argmin(distances, dim=1)
            sums.index_add_(0, assignment, batch)
            counts.index_add_(0, assignment, torch.ones_like(assignment, dtype=torch.float32))
        nonempty = counts > 0
        if bool(nonempty.any()):
            centers[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(1)
        empty = torch.nonzero(~nonempty, as_tuple=False).squeeze(1)
        if empty.numel() > 0:
            refill_idx = torch.randint(n_samples, (int(empty.numel()),), generator=generator, dtype=torch.long)
            centers[empty] = samples[refill_idx]
    return centers.clamp(0.0, 1.0)


def encode_patches_with_codebook(patches: torch.Tensor, codebook: torch.Tensor, *, batch_size: int = 4096) -> torch.Tensor:
    samples = torch.as_tensor(patches, dtype=torch.float32, device="cpu")
    centers = torch.as_tensor(codebook, dtype=torch.float32, device="cpu")
    assignments: list[torch.Tensor] = []
    for start in range(0, int(samples.shape[0]), max(1, int(batch_size))):
        batch = samples[start : start + int(batch_size)]
        distances = torch.cdist(batch, centers)
        assignments.append(torch.argmin(distances, dim=1))
    return torch.cat(assignments, dim=0)


@dataclass(slots=True)
class ImageTokenBenchmark:
    name: str
    data_dir: Path
    context_length: int
    vocab_size: int
    patch_size: int
    image_size: int
    codebook: torch.Tensor | None
    train_tokens: torch.Tensor
    val_tokens: torch.Tensor
    test_tokens: torch.Tensor
    train_images: torch.Tensor | None = None
    val_images: torch.Tensor | None = None
    test_images: torch.Tensor | None = None
    tokenization: str = "patch_codebook"
    dimensions_per_token: int = 48
    train_random_horizontal_flip: bool = False
    train_random_horizontal_flip_prob: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    def split_tokens(self, split: str) -> torch.Tensor:
        split_name = str(split).lower()
        if split_name == "train":
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
        if include_next_token:
            raise ValueError("Image-token benchmark does not support include_next_token=True.")
        starts = torch.randint(int(self.train_tokens.shape[0]), (int(batch_size),), generator=generator, dtype=torch.long)
        tokens = self.train_tokens[starts]
        if self.train_random_horizontal_flip:
            tokens = self._random_horizontal_flip_tokens(tokens, generator=generator)
        return tokens.to(device=device, dtype=torch.long, non_blocking=True)

    def _random_horizontal_flip_tokens(
        self,
        tokens: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        prob = float(self.train_random_horizontal_flip_prob)
        if prob <= 0.0:
            return tokens
        if prob > 1.0:
            raise ValueError(f"train_random_horizontal_flip_prob must be <= 1.0, got {prob}.")
        batch_size = int(tokens.shape[0])
        flip_mask = torch.rand((batch_size,), generator=generator, dtype=torch.float32) < prob
        if not bool(flip_mask.any()):
            return tokens
        flipped = tokens.clone()
        if str(self.tokenization).lower() == "rgb8":
            image_tokens = flipped.view(batch_size, int(self.image_size), int(self.image_size), 3)
            image_tokens[flip_mask] = torch.flip(image_tokens[flip_mask], dims=(2,))
            return flipped
        grid_size = int(self.image_size) // int(self.patch_size)
        if int(tokens.shape[1]) != grid_size * grid_size:
            raise ValueError(
                "Horizontal flip for patch-codebook tokens expects sequence length "
                f"{grid_size * grid_size}, got {int(tokens.shape[1])}."
            )
        patch_tokens = flipped.view(batch_size, grid_size, grid_size)
        patch_tokens[flip_mask] = torch.flip(patch_tokens[flip_mask], dims=(2,))
        return flipped

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
        if include_next_token:
            raise ValueError("Image-token benchmark does not support include_next_token=True.")
        tokens = self.split_tokens(split)
        if int(num_shards) > 1:
            tokens = tokens[int(shard_rank) :: int(num_shards)]
        if max_batches is not None:
            tokens = tokens[: int(batch_size) * int(max_batches)]
        for start in range(0, int(tokens.shape[0]), int(batch_size)):
            batch = tokens[start : start + int(batch_size)]
            if int(batch.shape[0]) == 0:
                continue
            yield batch.to(device=device, dtype=torch.long, non_blocking=True)

    def token_counts(self) -> dict[str, int]:
        return {
            "train": int(self.train_tokens.numel()),
            "val": int(self.val_tokens.numel()),
            "test": int(self.test_tokens.numel()),
        }

    def sequence_counts(self) -> dict[str, int]:
        return {
            "train": int(self.train_tokens.shape[0]),
            "val": int(self.val_tokens.shape[0]),
            "test": int(self.test_tokens.shape[0]),
        }

    def split_images(self, split: str) -> torch.Tensor | None:
        split_name = str(split).lower()
        if split_name == "train":
            return self.train_images
        if split_name == "val":
            return self.val_images
        if split_name == "test":
            return self.test_images
        raise ValueError(f"Unsupported split {split!r}.")

    def benchmark_metadata(self) -> dict[str, Any]:
        return {
            "benchmark_name": self.name,
            "data_dir": str(self.data_dir),
            "context_length": int(self.context_length),
            "vocab_size": int(self.vocab_size),
            "token_unit": "patch_token",
            "image_size": int(self.image_size),
            "patch_size": int(self.patch_size),
            "grid_size": int(self.image_size // self.patch_size),
            "tokenization": str(self.tokenization),
            "dimensions_per_token": int(self.dimensions_per_token),
            "split_token_counts": self.token_counts(),
            "split_sequence_counts": self.sequence_counts(),
            **self.metadata,
        }

    def decode_token_sequences(self, tokens: torch.Tensor, *, mask_token_id: int | None = None) -> torch.Tensor:
        if str(self.tokenization) == "rgb8":
            return reconstruct_images_from_rgb8_tokens(
                tokens,
                image_size=self.image_size,
                mask_token_id=mask_token_id,
            )
        if self.codebook is None:
            raise ValueError("Patch-codebook decoding requires a non-null codebook.")
        return reconstruct_images_from_patches(
            tokens,
            codebook=self.codebook,
            patch_size=self.patch_size,
            image_size=self.image_size,
            mask_token_id=mask_token_id,
        )


def load_cifar10_image_benchmark(cfg, *, project_root: Path) -> ImageTokenBenchmark:
    data_dir = _resolve_path(project_root, str(cfg.data_dir))
    if data_dir is None:
        raise ValueError("benchmark.data_dir must be set for CIFAR-10 image tokens.")
    metadata_path = _resolve_path(data_dir, str(cfg.tokenizer_path))
    if metadata_path is None or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Tokenizer metadata not found at {metadata_path}. Run scripts/prepare_cifar10_rgb8_tokens.py first."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata_patch_size = int(metadata["patch_size"])
    metadata_context_length = int(metadata["context_length"])
    metadata_codebook_size = int(metadata["codebook_size"])
    if int(cfg.patch_size) != metadata_patch_size:
        raise ValueError(
            f"benchmark.patch_size={int(cfg.patch_size)} does not match tokenizer metadata patch_size={metadata_patch_size}."
        )
    if int(cfg.context_length) != metadata_context_length:
        raise ValueError(
            "benchmark.context_length="
            f"{int(cfg.context_length)} does not match tokenizer metadata context_length={metadata_context_length}."
        )
    if int(cfg.codebook_size) != metadata_codebook_size:
        raise ValueError(
            "benchmark.codebook_size="
            f"{int(cfg.codebook_size)} does not match tokenizer metadata codebook_size={metadata_codebook_size}."
        )
    train_tokens = _load_tensor(data_dir / str(cfg.files.train))
    val_tokens = _load_tensor(data_dir / str(cfg.files.val))
    test_tokens = _load_tensor(data_dir / str(cfg.files.test))
    tokenization = str(metadata.get("tokenization", "patch_codebook"))
    dimensions_per_token = int(metadata.get("dimensions_per_token", int(metadata_patch_size) * int(metadata_patch_size) * 3))
    codebook = None
    codebook_file = metadata.get("codebook_file")
    if codebook_file:
        codebook = torch.load(data_dir / str(codebook_file), map_location="cpu")
        if int(codebook.shape[0]) != metadata_codebook_size:
            raise ValueError(
                f"Codebook row count {int(codebook.shape[0])} does not match tokenizer metadata codebook_size={metadata_codebook_size}."
            )
    train_images = None
    val_images = None
    test_images = None
    train_images_file = metadata.get("train_images_file")
    val_images_file = metadata.get("val_images_file")
    test_images_file = metadata.get("test_images_file")
    if train_images_file:
        train_images = torch.load(data_dir / str(train_images_file), map_location="cpu")
    if val_images_file:
        val_images = torch.load(data_dir / str(val_images_file), map_location="cpu")
    if test_images_file:
        test_images = torch.load(data_dir / str(test_images_file), map_location="cpu")
    augmentation_cfg = _cfg_get(cfg, "augmentation", None)
    train_random_horizontal_flip = False
    train_random_horizontal_flip_prob = 0.5
    if augmentation_cfg is not None:
        train_random_horizontal_flip = bool(_cfg_get(augmentation_cfg, "random_horizontal_flip", False))
        train_random_horizontal_flip_prob = float(_cfg_get(augmentation_cfg, "random_horizontal_flip_prob", 0.5))
    return ImageTokenBenchmark(
        name=str(cfg.name),
        data_dir=data_dir,
        context_length=metadata_context_length,
        vocab_size=metadata_codebook_size,
        patch_size=metadata_patch_size,
        image_size=int(metadata["image_size"]),
        codebook=None if codebook is None else torch.as_tensor(codebook, dtype=torch.float32),
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        test_tokens=test_tokens,
        train_images=None if train_images is None else torch.as_tensor(train_images, dtype=torch.float32),
        val_images=None if val_images is None else torch.as_tensor(val_images, dtype=torch.float32),
        test_images=None if test_images is None else torch.as_tensor(test_images, dtype=torch.float32),
        tokenization=tokenization,
        dimensions_per_token=dimensions_per_token,
        train_random_horizontal_flip=train_random_horizontal_flip,
        train_random_horizontal_flip_prob=train_random_horizontal_flip_prob,
        metadata=metadata,
    )
