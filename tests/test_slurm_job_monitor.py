from __future__ import annotations

import sys
import unittest
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from experiments.common.slurm_job_monitor import (
    build_resume_command,
    extract_wandb_id_from_names,
    normalize_slurm_state,
    parse_resume_artifacts,
    parse_sacct_output,
    parse_sbatch_submission,
    parse_squeue_output,
)


class SlurmJobMonitorTest(unittest.TestCase):
    def test_normalize_slurm_state_strips_suffix(self) -> None:
        self.assertEqual(normalize_slurm_state("FAILED+"), "FAILED")
        self.assertEqual(normalize_slurm_state("CANCELLED by 0"), "CANCELLED")

    def test_parse_squeue_output(self) -> None:
        rows = parse_squeue_output("45918061|RUNNING|None\n")
        self.assertEqual(rows["45918061"]["state"], "RUNNING")
        self.assertEqual(rows["45918061"]["reason"], "None")

    def test_parse_sacct_output_skips_steps(self) -> None:
        rows = parse_sacct_output(
            "45918061|FAILED|1:0|1:0|None\n"
            "45918061.batch|FAILED|1:0|1:0|None\n"
        )
        self.assertEqual(list(rows), ["45918061"])
        self.assertEqual(rows["45918061"]["state"], "FAILED")

    def test_parse_resume_artifacts(self) -> None:
        checkpoint, wandb_id = parse_resume_artifacts(
            "checkpoint=/cluster/jobs/run/checkpoints/model_step_0010000.pt\nwandb_id=abc123\n"
        )
        self.assertEqual(checkpoint, "/cluster/jobs/run/checkpoints/model_step_0010000.pt")
        self.assertEqual(wandb_id, "abc123")

    def test_extract_wandb_id_from_names(self) -> None:
        self.assertEqual(extract_wandb_id_from_names(["run-20260423_120000-abc123"]), "abc123")
        self.assertEqual(extract_wandb_id_from_names(["run-20260423_120000-abc123.wandb"]), "abc123")

    def test_parse_sbatch_submission(self) -> None:
        self.assertEqual(parse_sbatch_submission("Submitted batch job 45999999\n"), "45999999")

    def test_build_resume_command_structured_sbatch(self) -> None:
        command = build_resume_command(
            {
                "name": "demo",
                "run_dir": "/cluster/jobs/results/lm/exit/experiments/text8/exit/my_run",
                "submit_workdir": "/path/to/gexit-diffusion/code",
                "sbatch_script": "experiments/sweep_lm_exit_h200.sbatch",
                "submit_env": {
                    "CONFIG_NAME": "experiments/text8/exit",
                    "RESUME_CKPT": "{resume_ckpt}",
                    "RESUME_RUN_ID": "{resume_run_id}",
                    "WANDB_RESUME_ID": "{wandb_resume_id}",
                },
            },
            checkpoint="/cluster/jobs/results/lm/exit/experiments/text8/exit/my_run/checkpoints/model_step_0005000.pt",
            wandb_id="abc123",
        )
        self.assertIn("CONFIG_NAME=experiments/text8/exit", command)
        self.assertIn("RESUME_RUN_ID=my_run", command)
        self.assertIn("WANDB_RESUME_ID=abc123", command)


if __name__ == "__main__":
    unittest.main()
