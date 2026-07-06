# Server Evaluation Launchers

This directory keeps publication-scale evaluation launchers with private machine paths removed.

## Slurm

The `slurm/` scripts are Slurm wrappers for GPU-cluster runs:

- `run_text8_three_eval_same_gpu_h200hp.sbatch`: evaluates text8 EXIT, QSC, and I-MMSE checkpoints with `scripts/eval_lm_checkpoints.py`.
- `run_cifar10_three_eval_same_gpu_h200hp.sbatch`: runs CIFAR-10 image checkpoint evaluations for EXIT, I-MMSE, and QSC on one H200.

Set `PROJECT_DIR`, run directories, checkpoint paths, Slurm account/partition values, and W&B groups for your own cluster before using them.

## Multi-GPU VM

The `multigpu/` scripts are standalone multi-GPU VM launchers:

- `eval_qsc_d3pm_corrected_50k_ddp.py`: DDP QSC/D3PM-corrected 50k image generation and metrics.
- `eval_sde_bridge_equalinfo_50k_ddp.py`: DDP SDE bridge 50k image generation using equal-information SNR schedules.
- `eval_sde_bridge_derivative_50k_ddp.py`: DDP SDE bridge 50k image generation using derivative-based SNR schedules.

These scripts use placeholder paths such as `/path/to/gexit-diffusion`. Replace them with the target workspace and virtual-environment paths before running.
