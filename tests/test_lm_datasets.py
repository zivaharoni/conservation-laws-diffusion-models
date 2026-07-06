from __future__ import annotations

from array import array
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from omegaconf import OmegaConf

from scripts.download_lm_datasets import APP_ROOT as LM_APP_ROOT
from scripts.download_lm_datasets import default_dataset_root, prepare_dataset, verify_dataset
from src.lm_data import load_lm_benchmark


class _FakeEncoding:
    n_vocab = 128

    def encode_ordinary(self, text: str) -> list[int]:
        return [((ord(ch) % 97) + 1) for ch in text]


class LMDatasetBootstrapTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.archives_dir = self.root / "archives"
        self.archives_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_root = self.root / "data" / "lm"
        self.code_root = self.root / "code"
        self.code_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_text8_archive(self) -> Path:
        archive_path = self.archives_dir / "text8.zip"
        payload = bytes((idx % 251 for idx in range(100)))
        with zipfile.ZipFile(archive_path, "w") as zf:
            zf.writestr("text8", payload)
        return archive_path

    def _write_token_file(self, path: Path, tokens: list[int]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(array("I", [int(token) for token in tokens]).tobytes())

    def test_prepare_text8_from_local_zip_and_skip_on_second_run(self) -> None:
        archive_path = self._write_text8_archive()
        spec = {
            "archive_name": "text8.zip",
            "urls": [archive_path.resolve().as_uri()],
            "raw_member_basename": "text8",
            "raw_output_name": "text8",
            "expected_raw_bytes": 100,
            "split_bytes": {"train": 80, "val": 10, "test": 10},
        }
        dataset_dir = self.dataset_root / "text8"
        prepared = prepare_dataset("text8", dataset_dir, spec, keep_archives=False)
        self.assertEqual(prepared["status"], "prepared")
        self.assertTrue((dataset_dir / "text8").is_file())
        self.assertTrue((dataset_dir / "train.bin").is_file())
        info = json.loads((dataset_dir / "dataset_info.json").read_text())
        self.assertEqual(info["split_bytes"], {"train": 80, "val": 10, "test": 10})

        skipped = prepare_dataset("text8", dataset_dir, spec, keep_archives=False)
        self.assertEqual(skipped["status"], "skip")
        verified = verify_dataset("text8", dataset_dir, spec)
        self.assertEqual(verified["raw_bytes"], 100)

    def test_default_dataset_root_matches_trainer_repo_root(self) -> None:
        repo_root = LM_APP_ROOT.parent if LM_APP_ROOT.name == "code" else LM_APP_ROOT
        self.assertEqual(default_dataset_root(), repo_root / "data" / "lm")

    @mock.patch("src.lm_tokenizers._get_tiktoken_encoding", return_value=_FakeEncoding())
    def test_prepare_lm1b_from_mocked_hf_rows_and_skip_on_second_run(self, _mock_encoding) -> None:
        spec = {
            "hf_dataset": "dummy/lm1b",
            "hf_config": "",
            "text_field": "text",
            "train_split": "train",
            "test_split": "test",
            "validation_rows": 2,
            "train_shard_tokens": 8,
            "train_output_dir": "train",
            "val_output_name": "val.tokens",
            "test_output_name": "test.tokens",
            "tokenizer": "gpt2_bpe",
        }
        train_rows = [
            {"text": "alpha"},
            {"text": "beta"},
            {"text": "gamma"},
            {"text": "delta"},
        ]
        test_rows = [
            {"text": "eval one"},
            {"text": "eval two"},
        ]
        dataset_dir = self.dataset_root / "lm1b"
        with mock.patch("scripts.download_lm_datasets._load_hf_split", side_effect=[train_rows, test_rows]):
            prepared = prepare_dataset("lm1b", dataset_dir, spec, keep_archives=False)
        self.assertEqual(prepared["status"], "prepared")
        self.assertTrue((dataset_dir / "train").is_dir())
        self.assertTrue((dataset_dir / "val.tokens").is_file())
        self.assertTrue((dataset_dir / "test.tokens").is_file())
        self.assertGreaterEqual(prepared["train_shard_count"], 1)
        self.assertGreater(prepared["split_token_counts"]["train"], 0)
        self.assertGreater(prepared["split_token_counts"]["val"], 0)
        self.assertGreater(prepared["split_token_counts"]["test"], 0)

        info = json.loads((dataset_dir / "dataset_info.json").read_text(encoding="utf-8"))
        self.assertEqual(info["hf_dataset"], "dummy/lm1b")
        self.assertEqual(info["tokenizer_name"], "gpt2_bpe")

        skipped = prepare_dataset("lm1b", dataset_dir, spec, keep_archives=False)
        self.assertEqual(skipped["status"], "skip")
        verified = verify_dataset("lm1b", dataset_dir, spec)
        self.assertEqual(verified["split_token_counts"], info["split_token_counts"])

    def test_verify_lm1b_rejects_missing_shards_and_empty_eval_files(self) -> None:
        spec = {
            "train_output_dir": "train",
            "val_output_name": "val.tokens",
            "test_output_name": "test.tokens",
            "hf_dataset": "dummy/lm1b",
            "hf_config": "",
        }
        dataset_dir = self.dataset_root / "broken_lm1b"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "train").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "dataset_info.json").write_text(
            json.dumps({"split_token_counts": {"train": 0, "val": 0, "test": 0}, "train_shard_count": 0}),
            encoding="utf-8",
        )
        with self.assertRaises(FileNotFoundError):
            verify_dataset("lm1b", dataset_dir, spec)

        self._write_token_file(dataset_dir / "train" / "shard-00000.tokens", [1, 2, 3, 4])
        self._write_token_file(dataset_dir / "val.tokens", [])
        self._write_token_file(dataset_dir / "test.tokens", [5, 6, 7])
        (dataset_dir / "dataset_info.json").write_text(
            json.dumps({"split_token_counts": {"train": 4, "val": 0, "test": 3}, "train_shard_count": 1}),
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            verify_dataset("lm1b", dataset_dir, spec)

    @mock.patch("src.lm_tokenizers._get_tiktoken_encoding", return_value=_FakeEncoding())
    def test_loader_works_with_repo_root_relative_default_paths(self, _mock_encoding) -> None:
        text8_archive = self._write_text8_archive()
        text8_spec = {
            "archive_name": "text8.zip",
            "urls": [text8_archive.resolve().as_uri()],
            "raw_member_basename": "text8",
            "raw_output_name": "text8",
            "expected_raw_bytes": 100,
            "split_bytes": {"train": 80, "val": 10, "test": 10},
        }
        prepare_dataset("text8", self.dataset_root / "text8", text8_spec, keep_archives=False)

        lm1b_spec = {
            "hf_dataset": "dummy/lm1b",
            "hf_config": "",
            "text_field": "text",
            "train_split": "train",
            "test_split": "test",
            "validation_rows": 2,
            "train_shard_tokens": 8,
            "train_output_dir": "train",
            "val_output_name": "val.tokens",
            "test_output_name": "test.tokens",
            "tokenizer": "gpt2_bpe",
        }
        with mock.patch(
            "scripts.download_lm_datasets._load_hf_split",
            side_effect=[
                [{"text": "alpha"}, {"text": "beta"}, {"text": "gamma"}, {"text": "delta"}],
                [{"text": "test row"}],
            ],
        ):
            prepare_dataset("lm1b", self.dataset_root / "lm1b", lm1b_spec, keep_archives=False)

        text8_cfg = OmegaConf.create(
            {
                "name": "text8",
                "data_dir": "data/lm/text8",
                "context_length": 8,
                "tokenizer": {"name": "byte"},
                "files": {"raw": "text8", "train": "train.bin", "val": "val.bin", "test": "test.bin"},
                "splits": {"train_bytes": 80, "val_bytes": 10, "test_bytes": 10},
            }
        )
        benchmark = load_lm_benchmark(text8_cfg, project_root=self.root)
        self.assertEqual(benchmark.token_counts()["train"], 80)

        lm1b_cfg = OmegaConf.create(
            {
                "name": "lm1b",
                "data_dir": "data/lm/lm1b",
                "context_length": 4,
                "tokenizer": {"name": "gpt2_bpe"},
                "files": {
                    "train_dir": "train",
                    "val": "val.tokens",
                    "test": "test.tokens",
                },
            }
        )
        lm1b = load_lm_benchmark(lm1b_cfg, project_root=self.root)
        self.assertGreater(lm1b.token_counts()["train"], 0)
        batch = lm1b.sample_train_batch(batch_size=2, device="cpu", include_next_token=True)
        self.assertEqual(tuple(batch.shape), (2, 5))
        eval_batches = list(
            lm1b.iter_eval_batches(split="val", batch_size=2, device="cpu", include_next_token=False, max_batches=1)
        )
        self.assertEqual(tuple(eval_batches[0].shape)[1], 4)

    @mock.patch("src.lm_tokenizers._get_tiktoken_encoding", return_value=_FakeEncoding())
    def test_missing_data_error_mentions_bootstrap_command(self, _mock_encoding) -> None:
        cfg = OmegaConf.create(
            {
                "name": "text8",
                "data_dir": "data/lm/text8",
                "context_length": 8,
                "tokenizer": {"name": "byte"},
                "files": {"raw": "text8", "train": "train.bin", "val": "val.bin", "test": "test.bin"},
                "splits": {"train_bytes": 80, "val_bytes": 10, "test_bytes": 10},
            }
        )
        with self.assertRaises(FileNotFoundError) as exc:
            load_lm_benchmark(cfg, project_root=self.root)
        self.assertIn("python code/scripts/download_lm_datasets.py --dataset text8", str(exc.exception))

        lm1b_cfg = OmegaConf.create(
            {
                "name": "lm1b",
                "data_dir": "data/lm/lm1b",
                "context_length": 8,
                "tokenizer": {"name": "gpt2_bpe"},
                "files": {"train_dir": "train", "val": "val.tokens", "test": "test.tokens"},
            }
        )
        with self.assertRaises(FileNotFoundError) as lm1b_exc:
            load_lm_benchmark(lm1b_cfg, project_root=self.root)
        self.assertIn("python code/scripts/download_lm_datasets.py --dataset lm1b", str(lm1b_exc.exception))

if __name__ == "__main__":
    unittest.main()
