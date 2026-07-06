from __future__ import annotations

from array import array
import csv
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from omegaconf import OmegaConf

from experiments.common.wandb_logger import _merge_tags
from scripts.lm_common import maybe_apply_lm_lr_schedule
from scripts.report_lm_results import main as report_main
from scripts.train_lm_causal_ce import _validate_cfg as validate_causal_cfg
from scripts.train_lm_causal_ce import main as causal_main
from scripts.train_lm_exit import _validate_cfg as validate_exit_cfg
from scripts.train_lm_exit import main as exit_main
from scripts.train_lm_immse import _validate_cfg as validate_immse_cfg
from scripts.train_lm_immse import main as immse_main
from scripts.train_lm_qsc import _validate_cfg as validate_qsc_cfg
from scripts.train_lm_qsc import main as qsc_main
from src.lm_data import load_lm_benchmark
from src.lm_distributed import _resolve_device
from src.lm_models import CausalTransformerLM, ContinuousDiffusionTransformerLM, MaskedTransformerLM
from src.lm_objectives import compute_qsc_loss, evaluate_causal_nats, evaluate_exit_objective_metrics, evaluate_qsc_curve
from src.lm_tokenizers import build_tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _FakeEncoding:
    n_vocab = 128

    def encode_ordinary(self, text: str) -> list[int]:
        return [((ord(ch) % 97) + 1) for ch in text]


class LanguageModelingStackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_root = Path(self.temp_dir.name)
        self.text8_dir = self.data_root / "text8"
        self.text8_dir.mkdir(parents=True, exist_ok=True)
        raw = bytes((idx % 251 for idx in range(512)))
        (self.text8_dir / "text8").write_bytes(raw)
        self.lm1b_dir = self.data_root / "lm1b"
        (self.lm1b_dir / "train").mkdir(parents=True, exist_ok=True)
        self._write_token_file(self.lm1b_dir / "train" / "shard-00000.tokens", list(range(1, 65)))
        self._write_token_file(self.lm1b_dir / "val.tokens", list(range(65, 97)))
        self._write_token_file(self.lm1b_dir / "test.tokens", list(range(97, 128)))
        (self.lm1b_dir / "dataset_info.json").write_text(
            json.dumps(
                {
                    "hf_dataset": "dummy/lm1b",
                    "hf_config": "",
                    "hf_split_map": {"train": "train", "val": "train[:2]", "test": "test"},
                    "source_format": "huggingface_datasets",
                    "split_token_counts": {"train": 64, "val": 32, "test": 32},
                    "train_shard_count": 1,
                    "train_shard_target_tokens": 64,
                    "validation_rows": 2,
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_token_file(self, path: Path, tokens: list[int]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(array("I", [int(token) for token in tokens]).tobytes())

    def test_byte_tokenizer_and_eval_packing(self) -> None:
        benchmark_cfg = OmegaConf.create(
            {
                "name": "text8",
                "data_dir": str(self.text8_dir),
                "context_length": 16,
                "tokenizer": {"name": "byte"},
                "files": {"raw": "text8", "train": "train.bin", "val": "val.bin", "test": "test.bin"},
                "splits": {"train_bytes": 320, "val_bytes": 96, "test_bytes": 96},
            }
        )
        benchmark = load_lm_benchmark(benchmark_cfg, project_root=PROJECT_ROOT)
        self.assertEqual(benchmark.vocab_size, 256)
        batch = benchmark.sample_train_batch(batch_size=2, device="cpu", include_next_token=True)
        self.assertEqual(tuple(batch.shape), (2, 17))
        eval_batches = list(
            benchmark.iter_eval_batches(split="val", batch_size=2, device="cpu", include_next_token=False, max_batches=1)
        )
        self.assertEqual(tuple(eval_batches[0].shape), (2, 16))

    @mock.patch("src.lm_tokenizers._get_tiktoken_encoding", return_value=_FakeEncoding())
    def test_gpt2_bpe_tokenizer_uses_fake_encoding(self, _mock_encoding) -> None:
        tokenizer = build_tokenizer("gpt2_bpe")
        encoded = tokenizer.encode_text("abc")
        self.assertEqual(encoded, [1, 2, 3])
        self.assertEqual(tokenizer.metadata().vocab_size, 128)

    def test_causal_eval_reports_exact_nats_per_token(self) -> None:
        benchmark_cfg = OmegaConf.create(
            {
                "name": "text8",
                "data_dir": str(self.text8_dir),
                "context_length": 8,
                "tokenizer": {"name": "byte"},
                "files": {"raw": "text8", "train": "train.bin", "val": "train.bin", "test": "train.bin"},
                "splits": {"train_bytes": 320, "val_bytes": 96, "test_bytes": 96},
            }
        )
        benchmark = load_lm_benchmark(benchmark_cfg, project_root=PROJECT_ROOT)
        model = CausalTransformerLM(
            vocab_size=benchmark.vocab_size,
            hidden_size=16,
            n_layers=2,
            n_heads=4,
            ffn_hidden_size=32,
            max_seq_len=8,
            dropout=0.0,
        )
        metrics = evaluate_causal_nats(model, benchmark, split="val", batch_size=2, device="cpu", max_batches=1)
        self.assertIn("nats_per_token", metrics)
        self.assertGreater(metrics["evaluated_tokens"], 0)

    def test_lm_outputs_apply_optional_channel_tilt(self) -> None:
        model = MaskedTransformerLM(
            vocab_size=16,
            hidden_size=8,
            n_layers=2,
            n_heads=2,
            ffn_hidden_size=16,
            max_seq_len=8,
            dropout=0.0,
        )
        tokens = torch.tensor([[1, 2, 16, 3]], dtype=torch.long)
        outputs = model.forward_outputs(tokens)
        tilt = torch.full_like(outputs.logits_ext, 0.5)
        tilted = model.forward_outputs(tokens, channel_tilt=tilt)
        self.assertTrue(torch.allclose(outputs.logits, outputs.logits_ext))
        self.assertTrue(torch.allclose(tilted.logits, tilted.logits_ext + tilt))

    def test_diffusion_lm_returns_structured_outputs(self) -> None:
        model = ContinuousDiffusionTransformerLM(
            vocab_size=8,
            hidden_size=8,
            n_layers=2,
            n_heads=2,
            ffn_hidden_size=16,
            max_seq_len=6,
            dropout=0.0,
        )
        noisy = torch.randn(2, 4, 8)
        snr = torch.full((2, 4, 1), 0.25)
        outputs = model.forward_outputs(noisy, snr)
        self.assertEqual(tuple(outputs.hidden.shape), (2, 4, 8))
        self.assertEqual(tuple(outputs.emb_hat.shape), (2, 4, 8))
        self.assertEqual(tuple(outputs.logits_ext.shape), (2, 4, 8))
        self.assertEqual(tuple(outputs.logits.shape), (2, 4, 8))

    def test_exit_validation_steps_per_batch_repeats_noise_sampling(self) -> None:
        benchmark = self._tiny_text8_benchmark()
        model = MaskedTransformerLM(
            vocab_size=benchmark.vocab_size,
            hidden_size=16,
            n_layers=2,
            n_heads=4,
            ffn_hidden_size=32,
            max_seq_len=benchmark.context_length,
            dropout=0.0,
            use_timestep_conditioning=True,
        )
        single = evaluate_exit_objective_metrics(
            model,
            benchmark,
            split="val",
            batch_size=2,
            device="cpu",
            min_rate=0.1,
            max_rate=0.9,
            max_batches=1,
            steps_per_batch=1,
            seed=123,
        )
        repeated = evaluate_exit_objective_metrics(
            model,
            benchmark,
            split="val",
            batch_size=2,
            device="cpu",
            min_rate=0.1,
            max_rate=0.9,
            max_batches=1,
            steps_per_batch=3,
            seed=123,
        )
        self.assertEqual(single["evaluated_batches"], 1)
        self.assertEqual(repeated["evaluated_batches"], 3)
        self.assertEqual(single["evaluated_sequences"], 2)
        self.assertEqual(repeated["evaluated_sequences"], 6)
        self.assertTrue(torch.isfinite(torch.tensor(repeated["objective_loss_nats"])))

    def _tiny_text8_benchmark(self):
        benchmark_cfg = OmegaConf.create(
            {
                "name": "text8",
                "data_dir": str(self.text8_dir),
                "context_length": 8,
                "tokenizer": {"name": "byte"},
                "files": {"raw": "text8", "train": "train.bin", "val": "train.bin", "test": "train.bin"},
                "splits": {"train_bytes": 320, "val_bytes": 96, "test_bytes": 96},
            }
        )
        return load_lm_benchmark(benchmark_cfg, project_root=PROJECT_ROOT)

    def test_qsc_helpers_return_finite_metrics_and_curve(self) -> None:
        benchmark = self._tiny_text8_benchmark()
        model = MaskedTransformerLM(
            vocab_size=benchmark.vocab_size,
            hidden_size=16,
            n_layers=2,
            n_heads=4,
            ffn_hidden_size=32,
            max_seq_len=benchmark.context_length,
            dropout=0.0,
            use_timestep_conditioning=True,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(17)
        batch = benchmark.sample_train_batch(batch_size=2, device="cpu", include_next_token=False, generator=generator)
        t = torch.full((2,), 0.5, dtype=torch.float32)
        stats = compute_qsc_loss(model, tokens=batch, t=t, generator=generator)
        self.assertEqual(tuple(stats["losses"].shape), tuple(batch.shape))
        self.assertEqual(tuple(stats["observed_tokens"].shape), tuple(batch.shape))
        self.assertTrue(torch.isfinite(stats["loss"]))
        t_grid, curve_dict, metrics, _rows = evaluate_qsc_curve(
            model,
            benchmark,
            split="val",
            batch_size=2,
            device="cpu",
            t_min=0.1,
            t_max=0.9,
            n_t_points=5,
            max_batches=1,
            seed=123,
        )
        self.assertEqual(tuple(t_grid.shape), (5,))
        self.assertEqual(tuple(curve_dict["qsc_gexit"].shape), (5,))
        self.assertTrue(torch.isfinite(curve_dict["qsc_gexit"]).all())
        self.assertTrue(torch.isfinite(torch.tensor(metrics["curve_integral_nats"])))

    @mock.patch("src.lm_tokenizers._get_tiktoken_encoding", return_value=_FakeEncoding())
    def test_lm1b_loader_uses_sharded_train_split(self, _mock_encoding) -> None:
        benchmark_cfg = OmegaConf.create(
            {
                "name": "lm1b",
                "data_dir": str(self.lm1b_dir),
                "context_length": 8,
                "tokenizer": {"name": "gpt2_bpe"},
                "files": {"train_dir": "train", "val": "val.tokens", "test": "test.tokens"},
            }
        )
        benchmark = load_lm_benchmark(benchmark_cfg, project_root=PROJECT_ROOT)
        self.assertEqual(benchmark.token_counts()["train"], 64)
        batch = benchmark.sample_train_batch(batch_size=2, device="cpu", include_next_token=True)
        self.assertEqual(tuple(batch.shape), (2, 9))
        eval_batches = list(
            benchmark.iter_eval_batches(split="test", batch_size=2, device="cpu", include_next_token=False, max_batches=1)
        )
        self.assertEqual(tuple(eval_batches[0].shape), (2, 8))

    def test_lm_lr_schedule_warmup_and_cosine_decay(self) -> None:
        cfg = OmegaConf.create(
            {
                "train": {
                    "lr": 3.0e-4,
                    "n_steps": 4,
                    "lr_schedule": {
                        "enabled": True,
                        "warmup_steps": 1,
                        "min_lr": 1.0e-5,
                    },
                }
            }
        )
        model = torch.nn.Linear(4, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.train.lr))
        lrs = [maybe_apply_lm_lr_schedule(cfg, step=step, optimizer=optimizer) for step in range(1, 5)]
        self.assertAlmostEqual(lrs[0], 3.0e-4)
        self.assertLess(lrs[1], lrs[0])
        self.assertLess(lrs[2], lrs[1])
        self.assertAlmostEqual(lrs[3], 1.0e-5, places=10)

    def test_wandb_tag_merge_appends_publish_without_duplicates(self) -> None:
        tags = _merge_tags(["lm", "exit", "text8_exit"], "publish,lm")
        self.assertEqual(tags, ["lm", "exit", "text8_exit", "publish"])

    @mock.patch("torch.cuda.set_device")
    @mock.patch("torch.cuda.current_device", return_value=0)
    def test_resolve_device_indexes_single_process_cuda(self, _mock_current_device, mock_set_device) -> None:
        device = _resolve_device(requested_device="cuda", enabled=False, local_rank=0)
        self.assertEqual(str(device), "cuda:0")
        mock_set_device.assert_called_once_with(device)

    def _script_overrides(self, objective: str, benchmark_name: str, results_root: Path) -> list[str]:
        common = [
            f"paths.results_root={results_root}",
            "device=cpu",
            "benchmark.context_length=16",
            "model.hidden_size=32",
            "model.n_layers=2",
            "model.n_heads=4",
            "model.ffn_hidden_size=64",
            "logging.enabled=false",
            "train.n_steps=1",
            "train.save_every=1",
            "train.lr_schedule.enabled=true",
            "train.lr_schedule.warmup_steps=1",
            "train.lr_schedule.min_lr=1.0e-5",
            "logging.log_every=1",
            "eval.batch_size=2",
            "eval.max_batches=1",
        ]
        if objective == "causal":
            common.extend(
                [
                    "train.batch_size=2",
                    "train.grad_accum_steps=1",
                    "train.n_steps=2",
                    "eval.eval_every=1",
                ]
            )
        elif objective == "exit":
            common.extend(
                [
                    "train.batch_size=2",
                    "train.grad_accum_steps=1",
                    "train.n_steps=2",
                    "eval.eval_every=1",
                    "eval.n_s_points=5",
                ]
            )
        elif objective == "immse":
            common.extend(
                [
                    "train.batch_size=2",
                    "train.grad_accum_steps=1",
                    "train.n_steps=2",
                    "train.snr_min=1.0e-3",
                    "train.snr_max=1.0e-1",
                    "eval.eval_every=1",
                    "eval.n_snr_points=5",
                    "eval.dense_n_snr_points=7",
                    "eval.snr_min=1.0e-3",
                    "eval.snr_max=1.0e-1",
                ]
            )
        elif objective == "qsc":
            common.extend(
                [
                    "name=text8_qsc_test",
                    "logging.group=text8_qsc_test",
                    "train.batch_size=2",
                    "train.grad_accum_steps=1",
                    "train.n_steps=2",
                    "qsc.train_t_min=0.1",
                    "qsc.train_t_max=0.9",
                    "eval.eval_every=1",
                    "eval.n_t_points=5",
                    "eval.t_min=0.1",
                    "eval.t_max=0.9",
                ]
            )
        if benchmark_name == "text8":
            common.extend(
                [
                    f"benchmark.data_dir={self.text8_dir}",
                    "benchmark.splits.train_bytes=320",
                    "benchmark.splits.val_bytes=96",
                    "benchmark.splits.test_bytes=96",
                ]
            )
        elif benchmark_name == "lm1b":
            common.extend(
                [
                    "benchmark.name=lm1b",
                    f"benchmark.data_dir={self.lm1b_dir}",
                    "benchmark.tokenizer.name=gpt2_bpe",
                    "+benchmark.files.train_dir=train",
                    "benchmark.files.val=val.tokens",
                    "benchmark.files.test=test.tokens",
                ]
            )
        return common

    @mock.patch("src.lm_tokenizers._get_tiktoken_encoding", return_value=_FakeEncoding())
    def test_script_smoke_runs_and_reporting(self, _mock_encoding) -> None:
        results_root = self.data_root / "results"
        causal_main(["config_name=experiments/text8/causal_ce", *self._script_overrides("causal", "text8", results_root / "causal")])
        exit_main(["config_name=experiments/text8/exit", *self._script_overrides("exit", "text8", results_root / "exit")])
        immse_main(["config_name=experiments/text8/immse", *self._script_overrides("immse", "text8", results_root / "immse")])
        qsc_main(["config_name=experiments/text8/qsc", *self._script_overrides("qsc", "text8", results_root / "qsc")])
        causal_main(["config_name=experiments/text8/causal_ce", *self._script_overrides("causal", "lm1b", results_root / "causal_lm1b")])
        exit_main(["config_name=experiments/text8/exit", *self._script_overrides("exit", "lm1b", results_root / "exit_lm1b")])
        immse_main(["config_name=experiments/text8/immse", *self._script_overrides("immse", "lm1b", results_root / "immse_lm1b")])

        metric_files = sorted(results_root.rglob("metrics.json"))
        self.assertGreaterEqual(len(metric_files), 7)
        for path in metric_files:
            payload = json.loads(path.read_text())
            self.assertIn("ranking_metric_nats_per_token", payload)
            if payload.get("objective") == "causal_ce":
                self.assertIn("nats_per_token", payload["eval_details"]["val"])
            if payload.get("objective") == "exit":
                self.assertIn("objective_loss_nats", payload["eval_details"]["val"])
                self.assertIn("masked_ce_nats", payload["eval_details"]["val"])
                self.assertIn("unmasked_ce_nats", payload["eval_details"]["val"])
            if payload.get("objective") == "immse":
                self.assertIn("objective_loss_nats", payload["eval_details"]["val"])
                self.assertIn("mse", payload["eval_details"]["val"])
                self.assertIn("nll", payload["eval_details"]["val"])
                self.assertIn("nll_ext", payload["eval_details"]["val"])
            if payload.get("objective") == "qsc":
                self.assertIn("objective_loss_nats", payload["eval_details"]["val"])
                self.assertIn("ce_correct_nats", payload["eval_details"]["val"])
                self.assertIn("ce_incorrect_nats", payload["eval_details"]["val"])
        history_files = sorted(results_root.rglob("train_history.csv"))
        for path in history_files:
            with path.open() as handle:
                rows = list(csv.DictReader(handle))
            lrs = [float(row["lr"]) for row in rows if row.get("lr") not in ("", None)]
            embed_norms = [float(row["embed_norm"]) for row in rows if row.get("embed_norm") not in ("", None)]
            self.assertGreaterEqual(len(lrs), 1)
            self.assertGreaterEqual(len(embed_norms), 1)
            if "causal" in str(path):
                self.assertTrue(any(row.get("val_nats_per_token") not in ("", None) for row in rows))
                self.assertLess(lrs[-1], lrs[0])
            if "exit" in str(path):
                self.assertTrue(any(row.get("val_objective_loss_nats") not in ("", None) for row in rows))
                self.assertLess(lrs[-1], lrs[0])
            if "immse" in str(path):
                self.assertTrue(any(row.get("val_objective_loss_nats") not in ("", None) for row in rows))
                self.assertLess(lrs[-1], lrs[0])
            if "qsc" in str(path):
                self.assertTrue(any(row.get("val_objective_loss_nats") not in ("", None) for row in rows))
                self.assertLess(lrs[-1], lrs[0])
        literature_csv = self.data_root / "literature.csv"
        with literature_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["benchmark", "tokenizer", "model_name", "citation_key", "split", "nats_per_token", "bound_only_nats", "notes"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "benchmark": "text8",
                    "tokenizer": "gpt2_bpe",
                    "model_name": "external_diffusion",
                    "citation_key": "dummy2024",
                    "split": "test",
                    "nats_per_token": "",
                    "bound_only_nats": "3.14",
                    "notes": "bound only",
                }
            )

        output_csv = self.data_root / "summary.csv"
        output_tex = self.data_root / "summary.tex"
        report_main(
            [
                "--results-root",
                str(results_root),
                "--literature-csv",
                str(literature_csv),
                "--output-csv",
                str(output_csv),
                "--output-tex",
                str(output_tex),
            ]
        )
        self.assertTrue(output_csv.is_file())
        self.assertTrue(output_tex.is_file())

    def test_ddp_exit_large_smoke_run_writes_single_artifact_set(self) -> None:
        if not torch.distributed.is_available():
            self.skipTest("torch.distributed is not available")
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", 0))
                master_port = sock.getsockname()[1]
        except PermissionError:
            self.skipTest("sandbox does not permit opening localhost rendezvous sockets")
        results_root = self.data_root / "ddp_results"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nnodes=1",
                "--node-rank=0",
                "--master-addr=127.0.0.1",
                f"--master-port={master_port}",
                "--nproc_per_node=2",
                "scripts/train_lm_exit.py",
                "config_name=experiments/text8/exit",
                "device=cpu",
                f"paths.results_root={results_root}",
                f"benchmark.data_dir={self.text8_dir}",
                "benchmark.context_length=16",
                "benchmark.splits.train_bytes=320",
                "benchmark.splits.val_bytes=96",
                "benchmark.splits.test_bytes=96",
                "model.hidden_size=32",
                "model.n_layers=2",
                "model.n_heads=4",
                "model.ffn_hidden_size=64",
                "logging.enabled=false",
                "train.n_steps=2",
                "train.batch_size=2",
                "train.grad_accum_steps=1",
                "train.save_every=1",
                "train.lr_schedule.enabled=false",
                "logging.log_every=1",
                "eval.eval_every=1",
                "eval.batch_size=2",
                "eval.max_batches=1",
                "eval.n_s_points=3",
            ],
            cwd=str(PROJECT_ROOT),
            env={**os.environ, "OMP_NUM_THREADS": "1", "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(master_port)},
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(result.returncode, 0)
        metric_files = sorted(results_root.rglob("metrics.json"))
        history_files = sorted(results_root.rglob("train_history.csv"))
        curve_csvs = sorted(results_root.rglob("exit_curve.csv"))
        curve_plots = sorted(results_root.rglob("exit_curve.png"))
        models = sorted(results_root.rglob("model.pt"))
        self.assertEqual(len(metric_files), 1)
        self.assertEqual(len(history_files), 1)
        self.assertEqual(len(curve_csvs), 1)
        self.assertEqual(len(curve_plots), 1)
        self.assertEqual(len(models), 1)
        payload = json.loads(metric_files[0].read_text())
        self.assertEqual(payload["benchmark"]["benchmark_name"], "text8")
        self.assertEqual(payload["objective"], "exit")

    def test_sbatch_scripts_support_print_only(self) -> None:
        scripts = [
            PROJECT_ROOT / "experiments" / "sweep_lm_causal_ce_h200.sbatch",
            PROJECT_ROOT / "experiments" / "sweep_lm_exit_h200.sbatch",
            PROJECT_ROOT / "experiments" / "sweep_lm_immse_h200.sbatch",
            PROJECT_ROOT / "experiments" / "sweep_lm_qsc_h200.sbatch",
            PROJECT_ROOT / "experiments" / "bootstrap_lm_datasets.sbatch",
        ]
        for script in scripts:
            result = subprocess.run(
                ["bash", str(script)],
                cwd=str(PROJECT_ROOT),
                env={
                    **os.environ,
                    "PROJECT_DIR": str(PROJECT_ROOT),
                    "PRINT_ONLY": "true",
                    "CONDA_ENV": "gexit",
                },
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("arg=", result.stdout)


if __name__ == "__main__":
    unittest.main()
