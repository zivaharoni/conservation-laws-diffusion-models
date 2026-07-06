#!/bin/bash
#SBATCH --job-name=bootstrap_cifar10_rgb8_tokens
#SBATCH --account=research-gpu-account
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/path/to/gexit-diffusion/slurm/%x_%j.out
#SBATCH --error=/path/to/gexit-diffusion/slurm/%x_%j.err

set -euo pipefail

DEFAULT_PROJECT_DIR="/path/to/gexit-diffusion"
CODE_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-${DEFAULT_PROJECT_DIR}}}"
if [[ ! -f "${CODE_DIR}/scripts/prepare_cifar10_rgb8_tokens.py" && -f "${CODE_DIR}/code/scripts/prepare_cifar10_rgb8_tokens.py" ]]; then
  CODE_DIR="${CODE_DIR}/code"
fi
cd "${CODE_DIR}"

if ! command -v conda >/dev/null 2>&1 && [[ -f "${HOME}/.bashrc" ]]; then
  set +u
  source "${HOME}/.bashrc"
  set -u
fi

cmd=(conda run -n "${CONDA_ENV:-gexit}" python scripts/prepare_cifar10_rgb8_tokens.py)
cmd+=("--output-dir" "${OUTPUT_DIR:-data/img/cifar10_rgb8_tokens}")
cmd+=("--val-fraction" "${VAL_FRACTION:-0.02}")
cmd+=("--seed" "${SEED:-2026}")

printf '[bootstrap-cifar10-rgb8-tokens] arg=%s\n' "${cmd[@]}"
if [[ "${PRINT_ONLY:-false}" == "true" ]]; then
  exit 0
fi
"${cmd[@]}"
