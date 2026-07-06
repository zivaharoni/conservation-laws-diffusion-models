from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

from scripts.eval_img_generation_checkpoints import (
    build_image_model,
    checkpoint_step,
    discover_checkpoints,
    main as eval_main,
    sample_exit_images,
    sample_immse_images,
    sample_qsc_images,
)
from src.img_data import encode_rgb8_images, load_cifar10_image_benchmark


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ImageGenerationEvalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.data_dir = self.root / "rgb8"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(123)
        train_images = torch.randint(256, (6, 32, 32, 3), generator=generator, dtype=torch.uint8)
        val_images = torch.randint(256, (4, 32, 32, 3), generator=generator, dtype=torch.uint8)
        test_images = torch.randint(256, (4, 32, 32, 3), generator=generator, dtype=torch.uint8)
        torch.save(encode_rgb8_images(train_images), self.data_dir / "train_tokens.pt")
        torch.save(encode_rgb8_images(val_images), self.data_dir / "val_tokens.pt")
        torch.save(encode_rgb8_images(test_images), self.data_dir / "test_tokens.pt")
        torch.save(train_images.float() / 255.0, self.data_dir / "train_images.pt")
        torch.save(val_images.float() / 255.0, self.data_dir / "val_images.pt")
        torch.save(test_images.float() / 255.0, self.data_dir / "test_images.pt")
        (self.data_dir / "tokenizer_metadata.json").write_text(
            json.dumps(
                {
                    "benchmark_name": "cifar10_rgb8_tokens",
                    "tokenization": "rgb8",
                    "token_unit": "subpixel_token",
                    "image_size": 32,
                    "patch_size": 1,
                    "grid_size": 32,
                    "context_length": 3072,
                    "codebook_size": 256,
                    "dimensions_per_token": 1,
                    "color_channels": 3,
                    "train_images_file": "train_images.pt",
                    "val_images_file": "val_images.pt",
                    "test_images_file": "test_images.pt",
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _cfg(self, kind: str):
        model_name = "img_unet_rgb8_continuous" if kind == "img_immse" else "img_unet_rgb8"
        return OmegaConf.create(
            {
                "kind": kind,
                "objective": kind.removeprefix("img_"),
                "benchmark": {
                    "name": "cifar10_rgb8_tokens",
                    "data_dir": str(self.data_dir),
                    "tokenizer_path": "tokenizer_metadata.json",
                    "context_length": 3072,
                    "image_size": 32,
                    "patch_size": 1,
                    "codebook_size": 256,
                    "files": {"train": "train_tokens.pt", "val": "val_tokens.pt", "test": "test_tokens.pt"},
                },
                "model": {
                    "name": model_name,
                    "base_channels": 8,
                    "channel_multipliers": [1, 2, 2, 2],
                    "dropout": 0.0,
                    "time_embed_input_dim": 16,
                    "time_embed_dim": 32,
                    "input_skip": False,
                },
                "train": {"snr_min": 1.0e-6, "snr_max": 1.0},
                "eval": {"snr_min": 1.0e-6, "snr_max": 1.0, "t_min": 1.0 / 256.0, "t_max": 0.999},
                "qsc": {"train_t_min": 1.0 / 256.0, "train_t_max": 0.999},
                "logging": {"entity": "test-entity", "project": "test-project", "mode": "disabled"},
            }
        )

    def _benchmark(self):
        cfg = self._cfg("img_exit")
        return load_cifar10_image_benchmark(cfg.benchmark, project_root=PROJECT_ROOT)

    def test_checkpoint_discovery_sorts_by_step(self) -> None:
        ckpt_dir = self.root / "cache" / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        for name in ["model_step_0000010.pt", "model_step_0000002.pt", "model_step_0000005.pt"]:
            (ckpt_dir / name).write_bytes(b"x")
        paths = discover_checkpoints(self.root / "cache")
        self.assertEqual([checkpoint_step(path) for path in paths], [2, 5, 10])

    def test_channel_samplers_return_images(self) -> None:
        benchmark = self._benchmark()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(99)
        for kind in ("img_exit", "img_qsc", "img_immse"):
            cfg = self._cfg(kind)
            model = build_image_model(cfg, vocab_size=benchmark.vocab_size, context_length=benchmark.context_length).eval()
            if kind == "img_exit":
                images = sample_exit_images(
                    model,
                    benchmark,
                    count=2,
                    batch_size=1,
                    num_diff_steps=2,
                    temperature=1.0,
                    top_k=4,
                    device=torch.device("cpu"),
                    generator=generator,
                )
            elif kind == "img_qsc":
                images = sample_qsc_images(
                    model,
                    benchmark,
                    cfg=cfg,
                    count=2,
                    batch_size=1,
                    num_diff_steps=2,
                    temperature=1.0,
                    top_k=4,
                    device=torch.device("cpu"),
                    generator=generator,
                )
            else:
                images = sample_immse_images(
                    model,
                    benchmark,
                    cfg=cfg,
                    count=2,
                    batch_size=1,
                    num_diff_steps=2,
                    temperature=1.0,
                    top_k=4,
                    device=torch.device("cpu"),
                    generator=generator,
                )
            self.assertEqual(tuple(images.shape), (2, 32, 32, 3))
            self.assertGreaterEqual(float(images.min().item()), 0.0)
            self.assertLessEqual(float(images.max().item()), 1.0)

    def test_cli_uses_cached_files_without_wandb(self) -> None:
        cache_dir = self.root / "cache"
        output_dir = self.root / "eval"
        (cache_dir / "checkpoints").mkdir(parents=True)
        cfg = self._cfg("img_exit")
        OmegaConf.save(cfg, cache_dir / "config.yaml")
        benchmark = self._benchmark()
        model = build_image_model(cfg, vocab_size=benchmark.vocab_size, context_length=benchmark.context_length)
        torch.save({"step": 3, "model_state_dict": model.state_dict()}, cache_dir / "checkpoints" / "model_step_0000003.pt")
        exit_code = eval_main(
            [
                "--run",
                "offline-run",
                "--cache-dir",
                str(cache_dir),
                "--output-dir",
                str(output_dir),
                "--sample-count",
                "2",
                "--num-diff-steps",
                "2",
                "--batch-size",
                "1",
                "--top-k",
                "4",
                "--skip-metrics",
            ]
        )
        self.assertEqual(exit_code, 0)
        self.assertTrue((output_dir / "checkpoint_metrics.csv").is_file())
        payload = json.loads((output_dir / "checkpoint_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["checkpoints"][0]["step"], 3)


if __name__ == "__main__":
    unittest.main()
