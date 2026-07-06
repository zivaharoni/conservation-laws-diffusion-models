from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.img_common import nats_per_token_to_bpd
from scripts.lm_common import load_checkpoint_if_requested
from scripts.prepare_cifar10_rgb8_tokens import main as prepare_rgb8_main
from scripts.train_img_exit import main as img_exit_main
from scripts.train_img_immse import main as img_immse_main
from scripts.train_img_qsc import main as img_qsc_main
from src.img_data import (
    encode_rgb8_images,
    load_cifar10_image_benchmark,
    reconstruct_images_from_rgb8_tokens,
)
from src.img_models import CIFAR10DDPMUNet, RGB8ContinuousDDPMModel, RGB8ImageTokenUNet
from src.lm_objectives import compute_immse_loss


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent


class ImageStackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.data_dir = self.root / "cifar_tokens"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.rgb8_data_dir = self.root / "cifar_rgb8_tokens"
        self.rgb8_data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = self.root / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(7)
        train_tokens = torch.randint(16, (16, 64), generator=generator, dtype=torch.long)
        val_tokens = torch.randint(16, (8, 64), generator=generator, dtype=torch.long)
        test_tokens = torch.randint(16, (8, 64), generator=generator, dtype=torch.long)
        codebook = torch.rand((16, 4 * 4 * 3), generator=generator, dtype=torch.float32)
        patch_train_images = torch.rand((16, 32, 32, 3), generator=generator, dtype=torch.float32)
        patch_val_images = torch.rand((8, 32, 32, 3), generator=generator, dtype=torch.float32)
        patch_test_images = torch.rand((8, 32, 32, 3), generator=generator, dtype=torch.float32)
        torch.save(train_tokens, self.data_dir / "train_tokens.pt")
        torch.save(val_tokens, self.data_dir / "val_tokens.pt")
        torch.save(test_tokens, self.data_dir / "test_tokens.pt")
        torch.save(codebook, self.data_dir / "codebook.pt")
        torch.save(patch_train_images, self.data_dir / "train_images.pt")
        torch.save(patch_val_images, self.data_dir / "val_images.pt")
        torch.save(patch_test_images, self.data_dir / "test_images.pt")
        rgb8_images = torch.randint(256, (16, 32, 32, 3), generator=generator, dtype=torch.uint8)
        rgb8_val_images = torch.randint(256, (8, 32, 32, 3), generator=generator, dtype=torch.uint8)
        rgb8_test_images = torch.randint(256, (8, 32, 32, 3), generator=generator, dtype=torch.uint8)
        torch.save(encode_rgb8_images(rgb8_images), self.rgb8_data_dir / "train_tokens.pt")
        torch.save(encode_rgb8_images(rgb8_val_images), self.rgb8_data_dir / "val_tokens.pt")
        torch.save(encode_rgb8_images(rgb8_test_images), self.rgb8_data_dir / "test_tokens.pt")
        torch.save(rgb8_images.to(dtype=torch.float32) / 255.0, self.rgb8_data_dir / "train_images.pt")
        torch.save(rgb8_val_images.to(dtype=torch.float32) / 255.0, self.rgb8_data_dir / "val_images.pt")
        torch.save(rgb8_test_images.to(dtype=torch.float32) / 255.0, self.rgb8_data_dir / "test_images.pt")
        (self.data_dir / "tokenizer_metadata.json").write_text(
            json.dumps(
                {
                    "benchmark_name": "cifar10_image_tokens",
                    "image_size": 32,
                    "patch_size": 4,
                    "grid_size": 8,
                    "context_length": 64,
                    "codebook_size": 16,
                    "codebook_file": "codebook.pt",
                    "train_images_file": "train_images.pt",
                    "val_images_file": "val_images.pt",
                    "test_images_file": "test_images.pt",
                }
            ),
            encoding="utf-8",
        )
        (self.rgb8_data_dir / "tokenizer_metadata.json").write_text(
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

    def _benchmark_cfg(self, **overrides):
        values = {
            "name": "cifar10_image_tokens",
            "data_dir": str(self.data_dir),
            "tokenizer_path": "tokenizer_metadata.json",
            "context_length": 64,
            "patch_size": 4,
            "codebook_size": 16,
            "files": type("Files", (), {"train": "train_tokens.pt", "val": "val_tokens.pt", "test": "test_tokens.pt"})(),
        }
        values.update(overrides)
        return type("Cfg", (), values)()

    def _rgb8_benchmark_cfg(self, **overrides):
        values = {
            "name": "cifar10_rgb8_tokens",
            "data_dir": str(self.rgb8_data_dir),
            "tokenizer_path": "tokenizer_metadata.json",
            "context_length": 3072,
            "patch_size": 1,
            "codebook_size": 256,
            "files": type("Files", (), {"train": "train_tokens.pt", "val": "val_tokens.pt", "test": "test_tokens.pt"})(),
        }
        values.update(overrides)
        return type("Cfg", (), values)()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_rgb8_encode_and_reconstruct(self) -> None:
        rgb8_tokens = encode_rgb8_images(torch.randint(256, (2, 32, 32, 3), dtype=torch.uint8))
        rgb8_recon = reconstruct_images_from_rgb8_tokens(rgb8_tokens, image_size=32)
        self.assertEqual(tuple(rgb8_recon.shape), (2, 32, 32, 3))

    def test_rgb8_train_random_horizontal_flip(self) -> None:
        image = torch.arange(32 * 32 * 3, dtype=torch.long).view(1, 32, 32, 3) % 256
        train_tokens = encode_rgb8_images(image.repeat(4, 1, 1, 1).to(dtype=torch.uint8))
        torch.save(train_tokens, self.rgb8_data_dir / "train_tokens.pt")
        augmentation = type("Augmentation", (), {"random_horizontal_flip": True, "random_horizontal_flip_prob": 1.0})()
        benchmark = load_cifar10_image_benchmark(
            self._rgb8_benchmark_cfg(augmentation=augmentation),
            project_root=PROJECT_ROOT,
        )
        batch = benchmark.sample_train_batch(batch_size=2, device="cpu", include_next_token=False)
        flipped = batch.view(2, 32, 32, 3)
        expected = torch.flip(image, dims=(2,)).repeat(2, 1, 1, 1)
        self.assertTrue(torch.equal(flipped, expected))

    def test_benchmark_loader_and_models(self) -> None:
        cfg = self._benchmark_cfg()
        benchmark = load_cifar10_image_benchmark(cfg, project_root=PROJECT_ROOT)
        self.assertEqual(benchmark.vocab_size, 16)
        batch = benchmark.sample_train_batch(batch_size=3, device="cpu", include_next_token=False)
        self.assertEqual(tuple(batch.shape), (3, 64))

        ddpm_model = CIFAR10DDPMUNet(base_channels=32, channel_multipliers=(1, 2, 2, 2), time_embed_dim=128)
        ddpm_out = ddpm_model(torch.randn(2, 3, 32, 32), torch.tensor([1, 10]))
        self.assertEqual(tuple(ddpm_out.shape), (2, 3, 32, 32))
        self.assertAlmostEqual(
            nats_per_token_to_bpd(benchmark=benchmark, nats_per_token=48.0 * math.log(2.0)),
            1.0,
            places=6,
        )
        rgb8_benchmark = load_cifar10_image_benchmark(self._rgb8_benchmark_cfg(), project_root=PROJECT_ROOT)
        rgb8_batch = rgb8_benchmark.sample_train_batch(batch_size=2, device="cpu", include_next_token=False)
        self.assertEqual(tuple(rgb8_batch.shape), (2, 3072))
        rgb8_unet = RGB8ImageTokenUNet(vocab_size=256, image_size=32, base_channels=32, channel_multipliers=(1, 2, 2, 2), time_embed_dim=128)
        rgb8_unet_logits = rgb8_unet(rgb8_batch, timesteps=torch.full((2,), 0.3))
        self.assertEqual(tuple(rgb8_unet_logits.shape), (2, 3072, 256))
        rgb8_skip_unet = RGB8ImageTokenUNet(vocab_size=256, image_size=32, base_channels=32, channel_multipliers=(1, 2, 2, 2), time_embed_dim=128, input_skip=True)
        rgb8_skip_logits = rgb8_skip_unet(rgb8_batch, timesteps=torch.full((2,), 0.3))
        self.assertEqual(tuple(rgb8_skip_logits.shape), (2, 3072, 256))
        zero_skip = rgb8_skip_unet._input_skip_logits(rgb8_batch, torch.full((2,), 0.3))
        self.assertTrue(torch.allclose(zero_skip, torch.zeros_like(zero_skip)))
        masked_rgb8_batch = rgb8_batch.clone()
        masked_rgb8_batch[:, 0] = rgb8_skip_unet.mask_token_id
        with torch.no_grad():
            rgb8_skip_unet.input_skip_logits.weight.fill_(0.25)
            rgb8_skip_unet.input_skip_gate.weight.fill_(0.001)
        skip_t_low = rgb8_skip_unet._input_skip_logits(masked_rgb8_batch, torch.full((2,), 0.1))
        skip_t_high = rgb8_skip_unet._input_skip_logits(masked_rgb8_batch, torch.full((2,), 0.9))
        self.assertTrue(torch.allclose(skip_t_low[:, 0], torch.zeros_like(skip_t_low[:, 0])))
        self.assertGreater(float(skip_t_low[:, 1:].abs().sum().item()), 0.0)
        self.assertFalse(torch.allclose(skip_t_low[:, 1:], skip_t_high[:, 1:]))
        rgb8_cont = RGB8ContinuousDDPMModel(vocab_size=256, image_size=32, base_channels=32, channel_multipliers=(1, 2, 2, 2), time_embed_dim=128)
        noisy_rgb8 = rgb8_cont.x_values(rgb8_batch)
        rgb8_snr = torch.full((2, 3072, 1), 0.2)
        emb_hat_rgb8, logits_ext_rgb8, logits_rgb8 = rgb8_cont(noisy_rgb8, rgb8_snr)
        self.assertEqual(tuple(rgb8_cont.x_values.weight.shape), (256, 1))
        self.assertFalse(rgb8_cont.x_values.weight.requires_grad)
        self.assertAlmostEqual(float(rgb8_cont.x_values.weight.min().item()), -32.0, places=6)
        self.assertAlmostEqual(float(rgb8_cont.x_values.weight.max().item()), 32.0, places=6)
        self.assertEqual(tuple(emb_hat_rgb8.shape), (2, 3072, 1))
        self.assertEqual(tuple(logits_ext_rgb8.shape), (2, 3072, 256))
        self.assertEqual(tuple(logits_rgb8.shape), (2, 3072, 256))
        self.assertTrue(torch.equal(rgb8_cont.dequantize_mean_to_rgb_tokens(rgb8_cont.x_values.weight).squeeze(-1), torch.arange(256)))
        for loss_name in ("nll", "mse_nll"):
            stats = compute_immse_loss(
                rgb8_cont,
                tokens=rgb8_batch,
                snr=rgb8_snr,
                log_snr_width=1.0,
                loss_name=loss_name,
                weight_loss_by_snr=False,
            )
            self.assertTrue(torch.isfinite(stats["loss"]))

        rgb8_skip_cont = RGB8ContinuousDDPMModel(vocab_size=256, image_size=32, base_channels=32, channel_multipliers=(1, 2, 2, 2), time_embed_dim=128, input_skip=True)
        old_payload_path = self.root / "old_rgb8_immse_checkpoint.pt"
        old_state = rgb8_skip_cont.state_dict()
        old_state["x_values.weight"] = torch.randn(256, 64)
        old_state["input_skip_logits.weight"] = torch.randn(256, 64)
        old_optimizer = torch.optim.AdamW(rgb8_skip_cont.parameters(), lr=1.0e-4)
        torch.save({"model_state_dict": old_state, "optimizer_state_dict": old_optimizer.state_dict(), "step": 7, "tokens_seen": 123}, old_payload_path)
        new_optimizer = torch.optim.AdamW(rgb8_skip_cont.parameters(), lr=1.0e-4)
        start_step, tokens_seen = load_checkpoint_if_requested(model=rgb8_skip_cont, optimizer=new_optimizer, checkpoint_path=old_payload_path, project_root=PROJECT_ROOT)
        self.assertEqual(start_step, 8)
        self.assertEqual(tokens_seen, 123)
        self.assertEqual(tuple(rgb8_skip_cont.x_values.weight.shape), (256, 1))
        self.assertAlmostEqual(
            nats_per_token_to_bpd(benchmark=rgb8_benchmark, nats_per_token=math.log(2.0)),
            1.0,
            places=6,
        )

    def test_benchmark_loader_rejects_metadata_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "patch_size"):
            load_cifar10_image_benchmark(self._benchmark_cfg(patch_size=2), project_root=PROJECT_ROOT)
        with self.assertRaisesRegex(ValueError, "context_length"):
            load_cifar10_image_benchmark(self._benchmark_cfg(context_length=16), project_root=PROJECT_ROOT)
        with self.assertRaisesRegex(ValueError, "codebook_size"):
            load_cifar10_image_benchmark(self._benchmark_cfg(codebook_size=8), project_root=PROJECT_ROOT)

    def test_training_script_smokes(self) -> None:
        rgb8_unet_common = [
            "benchmark.data_dir=" + str(self.rgb8_data_dir),
            "benchmark.codebook_size=256",
            "logging.enabled=false",
            "logging.mode=dryrun",
            "train.n_steps=1",
            "train.batch_size=1",
            "train.grad_accum_steps=1",
            "train.save_every=1",
            "eval.report_splits=[val]",
            "eval.max_batches=1",
            "eval.batch_size=1",
            "eval.steps_per_batch=1",
            "model.base_channels=32",
            "model.time_embed_dim=128",
            "paths.results_root=" + str(self.results_dir / "rgb8_unet"),
        ]
        self.assertEqual(img_exit_main(["config_name=experiments/cifar10_rgb8_unet/exit", *rgb8_unet_common]), 0)
        self.assertEqual(img_qsc_main(["config_name=experiments/cifar10_rgb8_unet/qsc", *rgb8_unet_common]), 0)
        self.assertEqual(img_immse_main(["config_name=experiments/cifar10_rgb8_unet/immse", *rgb8_unet_common]), 0)
        rgb8_unet_metric_paths = sorted((self.results_dir / "rgb8_unet").rglob("metrics.json"))
        self.assertEqual(len(rgb8_unet_metric_paths), 3)

    def test_repo_root_print_only_launchers(self) -> None:
        launcher_names = [
            "sweep_img_exit_unet_h200.sbatch",
            "sweep_img_immse_unet_h200.sbatch",
            "sweep_img_qsc_unet_h200.sbatch",
        ]
        for name in launcher_names:
            completed = subprocess.run(
                ["bash", str(PROJECT_ROOT / "experiments" / name)],
                cwd=str(REPO_ROOT),
                env={
                    **dict(os.environ),
                    "PRINT_ONLY": "true",
                    "PROJECT_DIR": str(REPO_ROOT),
                    "DATA_DIR_OVERRIDE": str(self.rgb8_data_dir),
                    "WANDB_GROUP": "test_group",
                    "EXTRA_OVERRIDES": "model.base_channels=32 model.time_embed_dim=128",
                },
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("config_name=", completed.stdout)
            self.assertIn("benchmark.data_dir=", completed.stdout)

    def test_prepare_script_with_mock_dataset(self) -> None:
        class _FakeSplit(list):
            pass

        class _FakeImage:
            def __init__(self, value: int) -> None:
                self.value = value
                self.height = 32
                self.width = 32

            def convert(self, _mode: str):
                return self

            def tobytes(self) -> bytes:
                return bytes([self.value] * (32 * 32 * 3))

        def _fake_load_dataset(_dataset: str, split: str):
            count = 12 if split == "train" else 4
            return _FakeSplit([{"img": _FakeImage(idx % 255)} for idx in range(count)])

        import datasets

        original = datasets.load_dataset
        datasets.load_dataset = _fake_load_dataset
        try:
            rgb8_output_dir = self.root / "prepared_rgb8"
            rgb8_exit_code = prepare_rgb8_main(["--output-dir", str(rgb8_output_dir), "--val-fraction", "0.25"])
            self.assertEqual(rgb8_exit_code, 0)
            self.assertTrue((rgb8_output_dir / "train_tokens.pt").is_file())
            self.assertTrue((rgb8_output_dir / "tokenizer_metadata.json").is_file())
        finally:
            datasets.load_dataset = original


if __name__ == "__main__":
    unittest.main()
