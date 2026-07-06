from __future__ import annotations

import tempfile
import unittest
import csv
import json
from pathlib import Path

from experiments.common.config import compose_config
from experiments.common.runtime import build_data_spec, build_model
from scripts.export_paper_artifacts import main as export_paper_artifacts_main
from scripts.prepare_cifar10_rgb8_tokens import APP_ROOT as CIFAR_APP_ROOT
from scripts.prepare_cifar10_rgb8_tokens import resolve_output_dir
from scripts.run_repro_manifest import _command_for_profile, expanded_targets, load_manifest, selected_targets
from scripts.train_bsc import _validate_config as validate_bsc_config
from scripts.train_causal_ce import main as causal_ce_main
from scripts.train_immse import resolve_training_batching


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ReproducibilityEntrypointsTest(unittest.TestCase):
    def test_markov_qsc_config_composes_and_validates(self) -> None:
        cfg = compose_config(
            PROJECT_ROOT,
            [
                "config_name=experiments/markov_channel_k128/qsc",
                "logging.enabled=false",
                "train.n_steps=1",
                "train.batch_size=2",
                "train.seq_len=8",
                "eval.eval_batches=1",
                "eval.eval_batch_size=2",
                "eval.n_s_points=3",
                "model.embed_dim=8",
                "model.n_heads=2",
                "model.n_transformer_layers=1",
            ],
        )
        self.assertEqual(str(cfg.kind), "qsc")
        validate_bsc_config(cfg)

    def test_immse_model_uses_overridden_data_alphabet_size(self) -> None:
        cfg = compose_config(
            PROJECT_ROOT,
            [
                "config_name=experiments/markov_scaling/immse",
                "data.alphabet_size=8",
                "model.alphabet_size=128",
                "train.seq_len=8",
                "eval.seq_len=8",
                "model.embed_dim=8",
                "model.n_heads=2",
                "model.n_transformer_layers=2",
            ],
        )
        data_spec = build_data_spec(cfg)
        model = build_model(cfg, data_spec)
        self.assertEqual(model.alphabet_size, 8)

    def test_immse_microbatching_preserves_effective_batch_size(self) -> None:
        cfg = compose_config(
            PROJECT_ROOT,
            [
                "config_name=experiments/markov_scaling/immse",
                "train.batch_size=256",
                "train.grad_accum_steps=1",
                "train.microbatch_size=64",
            ],
        )
        batch_size, grad_accum_steps = resolve_training_batching(cfg)
        self.assertEqual(batch_size, 64)
        self.assertEqual(batch_size * grad_accum_steps, 256)

    def test_causal_ce_accepts_hydra_style_logging_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            status = causal_ce_main(
                [
                    "config_name=experiments/markov_scaling/causal_ce",
                    "logging.enabled=false",
                    "train.n_steps=1",
                    "train.batch_size=2",
                    "train.seq_len=8",
                    "eval.seq_len=8",
                    "data.alphabet_size=4",
                    "model.embed_dim=8",
                    "model.n_heads=2",
                    "model.n_transformer_layers=1",
                    "--results-root",
                    str(Path(tmp) / "results"),
                    "--eval-batches",
                    "1",
                    "--device",
                    "cpu",
                ]
            )
        self.assertEqual(status, 0)

    def test_cifar_relative_output_dir_matches_trainer_repo_root(self) -> None:
        repo_root = CIFAR_APP_ROOT.parent if CIFAR_APP_ROOT.name == "code" else CIFAR_APP_ROOT
        self.assertEqual(resolve_output_dir("data/img/cifar10_rgb8_tokens"), repo_root / "data" / "img" / "cifar10_rgb8_tokens")

    def test_repro_manifest_smoke_profile_selects_commands(self) -> None:
        manifest = load_manifest(PROJECT_ROOT / "experiments" / "repro" / "paper_commands.json")
        expanded_names = {target["name"] for target in expanded_targets(manifest)}
        self.assertNotIn("markov_qsc_k128", expanded_names)
        self.assertIn("markov_channel_qsc_k128", expanded_names)
        targets = selected_targets(manifest, names={"markov_channel_qsc_k128"}, groups=set())
        self.assertEqual([target["name"] for target in targets], ["markov_channel_qsc_k128"])
        command = _command_for_profile(manifest, targets[0], "smoke")
        self.assertIn("config_name=experiments/markov_channel_k128/qsc", command)
        self.assertIn("logging.enabled=false", command)
        self.assertIn("train.n_steps=2", command)

        bootstrap_targets = selected_targets(manifest, names={"bootstrap_text8"}, groups=set())
        bootstrap_command = _command_for_profile(manifest, bootstrap_targets[0], "smoke")
        self.assertNotIn("logging.enabled=false", bootstrap_command)

    def test_export_paper_artifacts_from_local_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "results"
            paper = root / "paper"
            self._write_markov_fixture(results, alphabet_size=8)
            self._write_conservation_fixture(results, family="text8", kind_prefix="lm", experiment_name_prefix="text8")
            self._write_conservation_fixture(results, family="cifar", kind_prefix="img", experiment_name_prefix="cifar10_rgb8_unet")

            status = export_paper_artifacts_main(
                [
                    "--results-root",
                    str(results),
                    "--paper-root",
                    str(paper),
                    "--alphabet-sizes",
                    "8",
                    "--allow-missing",
                    "--skip-cifar-suite-panels",
                ]
            )
            self.assertEqual(status, 0)

            figures = paper / "shared" / "figures"
            estimates = figures / "data" / "markov_synthetic" / "objective_estimates_d128.csv"
            self.assertTrue(estimates.is_file())
            with estimates.open("r", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["K"], "8")
            self.assertEqual(rows[0]["qsc"], "0.54")
            self.assertTrue((figures / "tikz" / "exp_text8_eval_curves.tex").is_file())
            self.assertTrue((figures / "tikz" / "exp_cifar_conservation_curves.tex").is_file())
            manifest = json.loads((figures / "data" / "paper_export_manifest.json").read_text())
            self.assertGreaterEqual(manifest["artifact_count"], 8)

    def test_export_allow_missing_removes_stale_conservation_tikz(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paper = root / "paper"
            full_results = root / "full_results"
            self._write_markov_fixture(full_results, alphabet_size=8)
            self._write_conservation_fixture(full_results, family="text8", kind_prefix="lm", experiment_name_prefix="text8")
            self._write_conservation_fixture(full_results, family="cifar", kind_prefix="img", experiment_name_prefix="cifar10_rgb8_unet")
            status = export_paper_artifacts_main(
                [
                    "--results-root",
                    str(full_results),
                    "--paper-root",
                    str(paper),
                    "--alphabet-sizes",
                    "8",
                    "--allow-missing",
                    "--skip-cifar-suite-panels",
                ]
            )
            self.assertEqual(status, 0)
            text8_tex = paper / "shared" / "figures" / "tikz" / "exp_text8_eval_curves.tex"
            self.assertTrue(text8_tex.is_file())

            partial_results = root / "partial_results"
            self._write_markov_fixture(partial_results, alphabet_size=8)
            self._write_conservation_fixture(partial_results, family="cifar", kind_prefix="img", experiment_name_prefix="cifar10_rgb8_unet")
            run_dir = partial_results / "lm" / "exit" / "text8_exit" / "run"
            self._write_json(run_dir / "metrics.json", {"experiment_kind": "lm_exit", "objective": "exit"})
            self._write_csv(
                run_dir / "exit_curve.csv",
                [{"s": 0.0, "masked_ce_nats": 0.9, "unmasked_ce_nats": 0.4, "exit_gap_nats": 0.5}],
            )
            status = export_paper_artifacts_main(
                [
                    "--results-root",
                    str(partial_results),
                    "--paper-root",
                    str(paper),
                    "--alphabet-sizes",
                    "8",
                    "--allow-missing",
                    "--skip-cifar-suite-panels",
                ]
            )
            self.assertEqual(status, 0)
            self.assertFalse(text8_tex.exists())
            latest = paper / "shared" / "figures" / "data" / "text8_conservation_curves" / "latest"
            self.assertTrue((latest / "exit_curve.csv").is_file())
            self.assertFalse((latest / "qsc_curve.csv").exists())

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload) + "\n")

    def _write_csv(self, path: Path, rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _write_markov_fixture(self, results: Path, *, alphabet_size: int) -> None:
        causal_dir = results / "causal_ce" / "markov" / "run"
        self._write_json(
            causal_dir / "metrics.json",
            {
                "family": "markov",
                "alphabet_size": alphabet_size,
                "reference_entropy_nats_per_token": 0.5,
                "eval_ce_nats_per_token": 0.51,
            },
        )
        exit_dir = results / "exit" / "markov_scaling_exit" / "run"
        self._write_json(
            exit_dir / "metrics.json",
            {
                "family": "markov",
                "alphabet_size": alphabet_size,
                "reference_entropy_nats": 0.5,
                "eval_exit_area_nats": 0.52,
            },
        )
        self._write_csv(
            exit_dir / "exit_curve.csv",
            [
                {"s": 0.0, "true_masked_ce": 0.0, "true_unmasked_ce": 0.0, "true_exit": 0.1, "model_masked_ce": 0.0, "model_unmasked_ce": 0.0, "model_exit": 0.2},
                {"s": 1.0, "true_masked_ce": 0.0, "true_unmasked_ce": 0.0, "true_exit": 0.3, "model_masked_ce": 0.0, "model_unmasked_ce": 0.0, "model_exit": 0.4},
            ],
        )
        immse_dir = results / "markov_scaling_immse" / "run"
        self._write_json(
            immse_dir / "metrics.json",
            {
                "alphabet_size": alphabet_size,
                "reference_entropy_nats": 0.5,
                "model_area_nats": 0.53,
            },
        )
        self._write_csv(
            immse_dir / "immse_curve.csv",
            [{"snr": 0.1, "true_mmse": 1.0, "model_mmse": 1.2, "true_half_mmse": 0.5, "model_half_mmse": 0.6}],
        )
        qsc_dir = results / "exit" / "markov_channel_qsc_k128" / f"experiments-markov-channel-k128-qsc__data-alphabet-size-{alphabet_size}"
        self._write_json(
            qsc_dir / "metrics.json",
            {
                "reference_entropy_nats": 0.5,
                "model_ce_area_nats": 0.54,
            },
        )
        self._write_csv(
            qsc_dir / "exit_curve.csv",
            [
                {"t": 0.25, "model_bsc_gexit": 0.7, "ref_bsc_gexit": 0.6},
                {"t": 0.75, "model_bsc_gexit": 0.8, "ref_bsc_gexit": 0.7},
            ],
        )

    def _write_conservation_fixture(self, results: Path, *, family: str, kind_prefix: str, experiment_name_prefix: str) -> None:
        base = results / ("lm" if family == "text8" else "img")
        for objective in ("exit", "qsc", "immse"):
            run_dir = base / objective / f"{experiment_name_prefix}_{objective}" / "run"
            self._write_json(
                run_dir / "metrics.json",
                {
                    "experiment_kind": f"{kind_prefix}_{objective}",
                    "objective": objective,
                    "run_id": f"{family}_{objective}_run",
                },
            )
        self._write_csv(
            base / "exit" / f"{experiment_name_prefix}_exit" / "run" / "exit_curve.csv",
            [{"s": 0.0, "masked_ce_nats": 1.0, "unmasked_ce_nats": 0.5, "exit_gap_nats": 0.5}, {"s": 1.0, "masked_ce_nats": 1.2, "unmasked_ce_nats": 0.4, "exit_gap_nats": 0.8}],
        )
        self._write_csv(
            base / "qsc" / f"{experiment_name_prefix}_qsc" / "run" / "qsc_curve.csv",
            [{"t": 0.25, "qsc_gexit_nats": 0.6}, {"t": 0.75, "qsc_gexit_nats": 0.9}],
        )
        self._write_csv(
            base / "immse" / f"{experiment_name_prefix}_immse" / "run" / "immse_curve.csv",
            [{"snr": 0.1, "mmse": 1.0, "half_mmse": 0.5}, {"snr": 1.0, "mmse": 0.4, "half_mmse": 0.2}],
        )


if __name__ == "__main__":
    unittest.main()
