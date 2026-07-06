#!/bin/bash
#SBATCH --job-name=lm_exit_h200
#SBATCH --account=research-gpu-account
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/path/to/gexit-diffusion/slurm/%x_%j.out
#SBATCH --error=/path/to/gexit-diffusion/slurm/%x_%j.err

set -euo pipefail

DEFAULT_PROJECT_DIR="/path/to/gexit-diffusion"
CODE_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-${DEFAULT_PROJECT_DIR}}}"
if [[ ! -f "${CODE_DIR}/scripts/train_lm_exit.py" && -f "${CODE_DIR}/code/scripts/train_lm_exit.py" ]]; then
  CODE_DIR="${CODE_DIR}/code"
fi
cd "${CODE_DIR}"

if [[ ! -f "scripts/train_lm_exit.py" ]]; then
  echo "Could not find scripts/train_lm_exit.py under CODE_DIR=${CODE_DIR}" >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1 && [[ -f "${HOME}/.bashrc" ]]; then
  set +u
  source "${HOME}/.bashrc"
  set -u
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available on PATH after shell initialization." >&2
  exit 1
fi

export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:+${WANDB_EXTRA_TAGS},}publish"

config_name="${CONFIG_NAME:-experiments/text8/exit}"
cmd=(conda run -n "${CONDA_ENV:-gexit}" python scripts/train_lm_exit.py "config_name=${config_name}")

if [[ -n "${BENCHMARK_OVERRIDE:-}" ]]; then
  cmd+=("benchmark=${BENCHMARK_OVERRIDE}")
fi
if [[ -n "${DATA_DIR_OVERRIDE:-}" ]]; then
  cmd+=("benchmark.data_dir=${DATA_DIR_OVERRIDE}")
fi
if [[ -n "${EXTRA_OVERRIDES:-}" ]]; then
  read -r -a extra_overrides <<< "${EXTRA_OVERRIDES}"
  cmd+=("${extra_overrides[@]}")
fi
if [[ -n "${RESUME_CKPT:-}" ]]; then
  if [[ -z "${RESUME_RUN_ID:-}" ]]; then
    echo "RESUME_RUN_ID must be set when RESUME_CKPT is provided." >&2
    exit 1
  fi
  cmd+=("train.resume_from=${RESUME_CKPT}" "logging.run_id=${RESUME_RUN_ID}")
  if [[ -n "${WANDB_RESUME_ID:-}" ]]; then
    cmd+=("logging.wandb_id=${WANDB_RESUME_ID}" "logging.resume=must")
  fi
fi

run_logs_dir="${RUN_LOGS_DIR:-${CODE_DIR}/results/slurm/lm_exit}"
mkdir -p "${run_logs_dir}"

printf '[lm-exit] arg=%s\n' "${cmd[@]}"
if [[ "${PRINT_ONLY:-false}" == "true" ]]; then
  exit 0
fi

"${cmd[@]}"
