#!/bin/bash
#SBATCH --job-name=bootstrap_lm_datasets
#SBATCH --account=research-gpu-account
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/path/to/gexit-diffusion/slurm/%x_%j.out
#SBATCH --error=/path/to/gexit-diffusion/slurm/%x_%j.err

set -euo pipefail

DEFAULT_PROJECT_DIR="/path/to/gexit-diffusion"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-${DEFAULT_PROJECT_DIR}}}"
SCRIPT_PATH=""
if [[ -f "${PROJECT_DIR}/scripts/download_lm_datasets.py" ]]; then
  SCRIPT_PATH="${PROJECT_DIR}/scripts/download_lm_datasets.py"
elif [[ -f "${PROJECT_DIR}/code/scripts/download_lm_datasets.py" ]]; then
  SCRIPT_PATH="${PROJECT_DIR}/code/scripts/download_lm_datasets.py"
fi

if [[ -z "${SCRIPT_PATH}" ]]; then
  echo "Could not find scripts/download_lm_datasets.py under PROJECT_DIR=${PROJECT_DIR}" >&2
  exit 1
fi

cd "${PROJECT_DIR}"

if ! command -v conda >/dev/null 2>&1 && [[ -f "${HOME}/.bashrc" ]]; then
  set +u
  source "${HOME}/.bashrc"
  set -u
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available on PATH after shell initialization." >&2
  exit 1
fi

dataset="${DATASET:-all}"
cmd=(conda run -n "${CONDA_ENV:-gexit}" python "${SCRIPT_PATH}" "--dataset" "${dataset}")

if [[ -n "${ROOT_OVERRIDE:-}" ]]; then
  cmd+=("--root" "${ROOT_OVERRIDE}")
fi
if [[ "${FORCE:-false}" == "true" ]]; then
  cmd+=("--force")
fi
if [[ "${VERIFY_ONLY:-false}" == "true" ]]; then
  cmd+=("--verify-only")
fi
if [[ "${KEEP_ARCHIVES:-false}" == "true" ]]; then
  cmd+=("--keep-archives")
fi

run_logs_dir="${RUN_LOGS_DIR:-${PROJECT_DIR}/results/slurm/bootstrap_lm_datasets}"
mkdir -p "${run_logs_dir}"

printf '[bootstrap-lm-datasets] arg=%s\n' "${cmd[@]}"
if [[ "${PRINT_ONLY:-false}" == "true" ]]; then
  exit 0
fi

"${cmd[@]}"
