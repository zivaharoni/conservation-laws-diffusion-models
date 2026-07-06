from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from src.img_data import encode_rgb8_images


def _repo_root() -> Path:
    return APP_ROOT.parent if APP_ROOT.name == "code" else APP_ROOT


def _load_split_arrays(dataset_name: str, split: str):
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    images = [_image_to_float_tensor(example["img"]) for example in ds]
    return torch.stack(images, dim=0)


def _image_to_float_tensor(image) -> torch.Tensor:
    rgb = image.convert("RGB")
    payload = torch.tensor(list(rgb.tobytes()), dtype=torch.uint8)
    return payload.view(rgb.height, rgb.width, 3).to(dtype=torch.float32) / 255.0


def _prepare_splits(train_images: torch.Tensor, test_images: torch.Tensor, *, val_fraction: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_count = int(train_images.shape[0])
    val_count = max(1, int(round(float(val_fraction) * train_count)))
    val_images = train_images[-val_count:].contiguous()
    new_train = train_images[:-val_count].contiguous()
    return new_train, val_images, test_images.contiguous()


def resolve_output_dir(value: str | Path) -> Path:
    output_dir = Path(value).expanduser()
    if not output_dir.is_absolute():
        output_dir = (_repo_root() / output_dir).resolve()
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare CIFAR-10 RGB8 subpixel tokens for comparable image-likelihood experiments.")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10"])
    parser.add_argument("--output-dir", default="data/img/cifar10_rgb8_tokens")
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)

    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_images_full = _load_split_arrays(args.dataset, "train")
    test_images = _load_split_arrays(args.dataset, "test")
    train_images, val_images, test_images = _prepare_splits(train_images_full, test_images, val_fraction=float(args.val_fraction))

    train_tokens = encode_rgb8_images(train_images)
    val_tokens = encode_rgb8_images(val_images)
    test_tokens = encode_rgb8_images(test_images)

    torch.save(train_tokens, output_dir / "train_tokens.pt")
    torch.save(val_tokens, output_dir / "val_tokens.pt")
    torch.save(test_tokens, output_dir / "test_tokens.pt")
    torch.save(train_images, output_dir / "train_images.pt")
    torch.save(val_images, output_dir / "val_images.pt")
    torch.save(test_images, output_dir / "test_images.pt")

    metadata = {
        "benchmark_name": "cifar10_rgb8_tokens",
        "dataset_name": str(args.dataset),
        "tokenization": "rgb8",
        "token_unit": "subpixel_token",
        "image_size": 32,
        "patch_size": 1,
        "grid_size": 32,
        "context_length": 32 * 32 * 3,
        "codebook_size": 256,
        "dimensions_per_token": 1,
        "color_channels": 3,
        "train_file": "train_tokens.pt",
        "val_file": "val_tokens.pt",
        "test_file": "test_tokens.pt",
        "train_images_file": "train_images.pt",
        "val_images_file": "val_images.pt",
        "test_images_file": "test_images.pt",
        "train_images": int(train_tokens.shape[0]),
        "val_images": int(val_tokens.shape[0]),
        "test_images": int(test_tokens.shape[0]),
        "seed": int(args.seed),
    }
    (output_dir / "tokenizer_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[prepare-cifar10-rgb8-tokens] wrote tokenized splits to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
